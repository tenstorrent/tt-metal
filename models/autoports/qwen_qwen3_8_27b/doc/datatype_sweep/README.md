# Qwen3.8-27B datatype sweep — stage 8

Selected **head_bfp4_lofi**: BFP4/LoFi decoder and LM-head matmuls,
BF16 projection activations/residuals/CCL, BFP8 KV cache, and FP32 destination
accumulation. Prefill top-1/top-5/top-100 are **100%/100%/100%**; traced
teacher-forcing decode is **99%/100%/100%**, **76.607 ms TTFT** and
**40.878 tokens/s/user** at B1/S203/G100.
Acceptance requires **top-1 ≥90%, top-5 ≥98%, top-100 =100%** for both paths.

Post-selection **token-out with no readback** is **41.095 tokens/s/user**
and **58.418 ms TTFT**, warmed B1/S128/G128 through the normal default
constructor. This is the token-out number for later reports and vLLM comparisons.
Complete deferred token delivery, including the final history transfer and list
construction, separately measures **41.075 tokens/s/user** and
**58.986 ms TTFT**. See [selected_token_out.json](selected_token_out.json).

## Selection and measured tradeoff

All 15 evaluated full64 policies pass accuracy. The selected policy is the fastest
measured traced teacher-forcing candidate, improving **1.83%**
over the policy-plumbed baseline in the same regime. BFP4 KV plus the selected head
is 40.857 tokens/s/user: its 0.05% difference is within ordinary run variation,
and it provides no measured speed advantage. Keeping BFP8 KV both wins the
observed ranking and preserves the simpler existing cache policy.

| Config | Decode top-1 | Top-5 / top-100 | Traced TF t/s/user | Decision |
| --- | ---: | ---: | ---: | --- |
| `head_bfp4_lofi` | 99% | 100% / 100% | 40.878 | Selected |
| `head_bfp4_lofi_kv_bfp4` | 98% | 100% / 100% | 40.857 | Slower; accuracy passed |
| `head_bfp4_hifi2` | 99% | 100% / 100% | 40.735 | Slower; accuracy passed |
| `head_bfp4_lofi_no_fp32_acc` | 99% | 100% / 100% | 40.717 | Slower; accuracy passed |
| `head_bfp8_lofi` | 100% | 100% / 100% | 40.182 | Slower; accuracy passed |
| `kv_bf16` | 98% | 100% / 100% | 40.159 | Slower; accuracy passed |
| `kv_bfp4` | 98% | 100% / 100% | 40.157 | Slower; accuracy passed |
| `baseline_bfp4_lofi_head_bfp8_hifi2` | 98% | 100% / 100% | 40.142 | Slower; accuracy passed |
| `ccl_bfp8` | 99% | 100% / 100% | 39.649 | Slower; accuracy passed |
| `activation_bfp8` | 100% | 100% / 100% | 39.174 | Slower; accuracy passed |
| `decoder_bfp4_hifi2` | 98% | 100% / 100% | 38.061 | Slower; accuracy passed |
| `inner_mlp_bfp4_lofi_outer_bfp8` | 98% | 100% / 100% | 36.160 | Slower; accuracy passed |
| `canonical_qwen36_mixed_lofi` | 100% | 100% / 100% | 34.350 | Slower; accuracy passed |
| `canonical_decoder_bfp8_lofi` | 100% | 100% / 100% | 30.718 | Slower; accuracy passed |
| `canonical_decoder_bfp8_hifi2` | 100% | 100% / 100% | 30.366 | Slower; accuracy passed |

The exact dtype/fidelity policies, layer exceptions, prefill accuracy, repeated
samples, TTFT, commands, hardware and source provenance are in
[sweep_results.json](sweep_results.json) and [sweep_results.csv](sweep_results.csv).
[matmul_group_results.csv](matmul_group_results.csv) expands every candidate by
material matmul group, including kept/rejected decisions and reasons.
[failed_attempts.json](failed_attempts.json) records interrupted or launcher-failed
attempts separately; neither failure is a numerical precision rejection.

Uniform BFP8/LoFi is faster than uniform BFP8/HiFi2, but both are substantially
slower than BFP4/LoFi. BFP4/HiFi2 decoder and head comparisons likewise give no
speed advantage. Every considered BFP4 group has full-model BFP4/LoFi evidence,
including the LM head. The closest canonical mapping and its differences from
this autoport are explicit in [canonical_policy_mapping.md](canonical_policy_mapping.md).
The first/last-layer exception candidate was measured; it was unnecessary for
accuracy and slower. BFP8 CCL and projection activations passed but were slower.
The combined head/KV candidate tests the compatible memory reductions together.
Disabling FP32 destination accumulation also passed, but measured only 40.717
tokens/s/user, so FP32 accumulation remains enabled.
The Stage 7 head LoFi component PCC rejection does not determine this stage's
selection: the required full-model top-k gates here pass with actual target weights.

## Pareto plots

![Top-1 accuracy and traced teacher-forcing performance](top1_perf_pareto.png)

![Top-5 accuracy and traced teacher-forcing performance](top5_perf_pareto.png)

Both charts plot all 15 evaluated full-model configs, connect the non-dominated
frontier, mark the selected point red, and show the minimum accuracy as a
vertical dotted line. Top-1 has a second frontier point, BFP8/LoFi head at 100%
accuracy and lower throughput. All configs have 100% top-5, so that frontier
reduces to the selected fastest point. Accuracy gates apply jointly; frontier
membership alone is insufficient. Coincident points retain separate numbered
entries with exact throughput in the plot key and CSV.

## Measurement contract and refreshed baseline

Hardware is four Blackhole p300c devices, MeshShape(1,4), TP4 Ring with 8192-byte
fabric payload and one pass-through host thread. The pinned main AIME24
chat-template reference has S203/G100, top-k 100, revision
`1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`. Each raw candidate records its
reference metadata. Batch size is one and all 64 layers execute.
The exact small oracle and metadata are preserved in [reference/](reference/),
byte-identical to the model-root refpt used by every run. On a fresh checkout,
copy those two files to the model root before reproducing the recorded commands.

The unchanged optimized full-model baseline was refreshed first:
[baseline_original.json](baseline_original.json), prefill 99%/100%/100%, decode
98%/100%/100%, warmed traced teacher forcing 40.216 tokens/s/user and 77.932 ms
TTFT. Its deferred complete-delivery S128/G128 control is 40.387 tokens/s/user,
58.846 ms TTFT. These two regimes are recorded separately. After policy plumbing,
the repeated baseline measures 40.142 traced TF tokens/s/user with identical
accuracy; that repeated baseline is the matrix comparison.

Ranking uses the median of the last two warmed teacher-forcing samples after a
separate warmup: 99 model trace replays, 99 sampling trace replays, reference-token
feedback and sampled-token delivery. No trace capture or full-logits transfer
occurs inside that ranked window. The [reference ordering audit](reference_topk_order_audit.json) reproduces all
available native-token accuracy arrays against the standard stored top-k metric;
one HF tied maximum differs between its greedy token and top-k first entry.
The standard readiness callback/capture timing
is retained for correctness provenance but is not the ranking metric. The normal
selected-default confirmation is separately reported and must reproduce within
2%; it is not used to replace candidate rows selectively.

The post-selection benchmark is the Stage 7 `benchmark_full_model.py --full`
loop unchanged except for recording the resolved precision policy. Its queued
row runs 127 model/sampling replays, one final synchronization, no host input
refresh and no per-token readback; it checks the final token outside timing.
Deferred complete delivery is a separate row and includes history appends and
one history transfer. Fixed S128 prompts are latency fixtures, not qualitative
judgments. Cold construction/capture time is not included in warmed metrics.

## Runtime construction and policy propagation

[selected_precision_config.json](selected_precision_config.json) is consumed by
`tt/precision.py`, shared by direct `QwenModel` construction and `build_generator`.
The precedence is explicit `precision_config` argument, `QWEN_PRECISION_CONFIG`,
selected artifact when present, then safe baseline. `QWEN_PRECISION_CONFIG=baseline`
restores the Stage 7 policy. The artifact includes all six weight groups,
per-group compute fidelities, FP32 accumulation, layer exceptions, activations,
residuals, CCL, KV, logits/sampling assumptions and maximum context.

`attention` maps to packed Q/K/V/gate or GDN input projections; `output` maps to
attention/GDN output. Packed MLP gate/up share dtype and fidelity; `down` is the
MLP down projection. The dense model has 48 linear-attention and 16 full-attention
layers; MoE routing does not apply. CCL dtype governs decoder collectives and
shared workspaces. Embedding and sampler candidate collectives retain their
BF16 contracts. Noncanonical layer-exception keys are rejected;
[policy_validation_equivalence.json](policy_validation_equivalence.json) proves
all measured configs resolve identically before and after that validation fix.
Native fixed assumptions (BF16 norms/embeddings/convolution,
FP32 recurrence, BF16 logits/sampling, uint32 tokens, sensitive HiFi4 and final
norm HiFi2) are validated before allocation; unsupported changes fail rather
than being silently ignored. Selected residual dtype is passed to residual adds.

Actual per-layer uploaded weight dtypes, DRAM copies, compute-kernel fields,
constructed policies, allocated cache/logit/token dtypes and trace counters are
recorded in [selected_confirmation.json](selected_confirmation.json).
[propagation_check.json](propagation_check.json) checks these against the selected
artifact and normal default, plus readiness/token-out policy equality. There is
no vLLM adapter in this stage; future adapters using the shared construction path
inherit this default. No vLLM integration was started.

## Capability and quality evidence

The selected shared-suite haiku has a **5/7/6** meter, versus 5/7/5 in the HF and
BFP8-head baseline controls. This retained quality limitation is reproduced by
selected native replay, host-greedy sampling and the BFP4/HiFi2 head control at
identical cache geometry; higher head fidelity does not correct it. See
[qualitative_review.md](qualitative_review.md) and [AUTOFIX_haiku.md](AUTOFIX_haiku.md).
The numeric acceptance gates and fastest traced performance determine selection;
the shared suite is not claimed universally instruction-correct.

[memory_head_bfp4_lofi.json](memory_head_bfp4_lofi.json) and the updated
[context contract](../context_contract.json) preserve the advertised **262144**
context. Conservative total is 13,219,443,200 bytes/device against physical
34,138,688,512 bytes, leaving 20,919,245,312 bytes. Every tested KV policy has its
own `memory_*.json` calculation. Tile sizes, actual inherited projection layouts,
selected head/KV and conservative scratch/history reserves are included. Memory
arithmetic is not an execution or maximum-context accuracy test.

The final evidence is:

- [selected_readiness.json](selected_readiness.json): pinned AIME24 checks,
  exact-prompt autoregressive controls, shared chat-template qualitative suite,
  full64 S262143/G2 and S262144/G1 context/position execution.
- [selected_non_aligned.json](selected_non_aligned.json): full64 logical prompt
  lengths 1/31/32/33/4095/4096/4097, trace/eager parity, physical-page remapping,
  retained-output lifetime and capture guards.
- [selected_contract_b32.json](selected_contract_b32.json): fixed slots 31/0,
  prompts 31/33, inactive cache preservation, remapping and continuation.
- [selected_watcher.json](selected_watcher.json): separate real-layer 0/3 watcher
  and trace-allocation tracking, Ethernet checking enabled, no device profiler.
- [qualitative_review.md](qualitative_review.md) and
  [qualitative_metrics.json](qualitative_metrics.json): direct output inspection,
  pinned HF controls, degeneracy and generated-code checks. Raw prompt IDs,
  rendered templates, TT/HF text and tokens remain in `tt_qualitative*.json`;
  the original controls and hashes are in `hf_control_provenance.json`.

Batch 32 short prompts and batch 1 maximum context are separate capabilities.
The 262144-token runs establish capacity/position behavior, not an HF accuracy
oracle over that length. The 100-position AIME24 agreement percentages are
readiness metrics, not an AIME dataset task score. These measurements identify
the fastest evaluated precision policy at the inherited optimized geometry;
they do not claim a global optimum or statistical confidence interval.

## Reproduction, review and provenance

[commands.log](commands.log) contains exact invocations. Typical reproduction:

```bash
QWEN_PRECISION_CONFIG=models/autoports/qwen_qwen3_8_27b/doc/datatype_sweep/configs/head_bfp4_lofi.json \
  bash models/autoports/qwen_qwen3_8_27b/tests/run_datatype_experiment.sh repeat_head \
  models.autoports.qwen_qwen3_8_27b.tests.run_datatype_candidate
env -u QWEN_PRECISION_CONFIG bash models/autoports/qwen_qwen3_8_27b/tests/run_datatype_experiment.sh repeat_token_out \
  models.autoports.qwen_qwen3_8_27b.tests.benchmark_full_model --full
```

Run all hardware-facing commands serially. Each run has JSON, console log, exit
status, base Git SHA, whitelisted environment and source/native hashes. Archived
`source_snapshots/` preserve the runtime sources; final runs also preserve test
sources and the selected precision artifact. The launcher environment comes from
the installed model-bringup package. The startup mesh/list evidence and recovered
upload stall are recorded in [work_log.md](work_log.md),
[AUTOTRIAGE_kv_startup.md](AUTOTRIAGE_kv_startup.md), and
[AUTOFIX_kv_startup.md](AUTOFIX_kv_startup.md). The successful identical-policy
retry establishes recovery; the native infrastructure root cause is unproven.

Only Python/runtime metadata and documentation changed; no C++ build is required.
[precommit.log](precommit.log), propagation/qualitative checks and the final
validation index record host verification. Independent review returned **clean-pass** in
[stage_review.md](stage_review.md), with the controlled haiku limitation retained.
Local checkpoint SHAs are recorded in
[work_log.md](work_log.md); no push is authorized or performed.
