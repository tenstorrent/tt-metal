# Datatype-sweep work log

## Contract and environment

- Date: 2026-08-12 UTC; start SHA `8cf41e9b384`.
- Accuracy gates: top-1 >= 90%, top-5 >= 98%; top-100 retained at 100%.
- Hardware: four Blackhole p300c, 1x4 `FABRIC_1D_RING`, TP4, two links.
- `tt-smi -ls --local` showed all four chips and the bounded mesh open/close
  smoke passed. Hardware commands were serialized; no Watcher/profiler mixing.
- The repo-local AIME24 reference was regenerated with the exact Falcon3
  snapshot, base-model completion formatting (tokenizer chat template is null),
  top-k 100, and 100 generated tokens: `results/reference/aime24_100.refpt`.
- Unrelated untracked `third_party/tt-metal/` was not touched.

## Baseline and selection

Default artifact propagation was added before ranking. The normal generator
requires `selected_precision_config.json`; all policy fields construct or
validate the real runtime path, and every evidence JSON emits a runtime
`precision_summary`. Baseline prefill is 92/100 top-1 and 100/100 top-5/top-100.
Teacher forcing is 93/100 top-1 and 100/100 top-5/top-100. The dedicated traced
regime measures 25.044 ms warmed TTFT and 110.815 teacher-forcing t/s/u.

Ten policies were attempted. Nine ran full accuracy and trace performance;
all nine passed. The isolated BF16 activation, CCL, and KV controls show only
BF16 KV improves traced performance. The fastest passing policy is BFP4+LoFi
with BF16 KV at 110.972 t/s/u. Its fresh normal-path separate token-out rerun is
`results/selected/post_selection_token_out.json`: 110.991 device-only and
110.403 caller-visible t/s/u at batch 1, prompt 128, 128 warmed
trace replays. This is the serving-comparison number for later stages.

## AutoFix and limitations

BFP8/BF16 initially failed in terminal LM-head construction because the BFP4
one-piece 32K-column geometry hard-coded `in0_block_w=3`. Fresh AutoDebug
localized the failure. AutoFix verified BFP8 at width 1 on real weights and
implemented the smallest dtype-aware geometry. BF16 remained physically
blocked at width 1 for 32K, 16K, and 8K splits with the identical 2,003,712-byte
CB requirement above 1,572,864-byte L1. See `AUTODEBUG.md`, `AUTOFIX.md`, and
`results/autofix_lm_head/`. This is a construction blocker, not an accuracy or
performance datapoint, and is excluded from Pareto plots.

Nanobind lifetime diagnostics recur after successful device close and metric
emission across all controls. They do not touch measured execution or device
health; prior optimized/full-model reviews classified the same shutdown-only
warning as controlled.

## Capability and qualitative gates

- `results/all_bfp4_lofi_bf16_kv/full_context_coverage.json`: selected BF16 KV policy passes
  the complete 32,768-token contract with all 28 layers/final pages.
- `results/all_bfp4_lofi_bf16_kv/non_aligned_contract.json`: 33/47 and 2049/2079 non-aligned
  prompts, inactive slots, page rounding, trace replay, reset, greedy, and
  stochastic paths pass.
- `results/selected_bf16_kv_qualitative/`: six 100-token TT/HF completion controls
  with prompt-format metadata; degeneracy checker passes. The repeated haiku
  stanza matches the previous full-model selected-policy control and has no
  corrupt-token symptom; classified in `qualitative_verdict.md`.

## Commands

Principal commands use strict fallback and were run once per candidate:

```text
python -m models.common.readiness_check.generate --hf-model <snapshot> --prompt-source aime24 --gen-len 100 --top-k 100 --output results/reference/aime24_100.refpt
FALCON3_PRECISION_CONFIG=<config> TTNN_CONFIG_OVERRIDES='{"throw_exception_on_fallback": true}' python -m models.common.readiness_check.run_teacher_forcing --model-dir models/autoports/tiiuae_falcon3_7b_base --reference results/reference/aime24_100.refpt --mesh-device P300X2 --fabric-config FABRIC_1D_RING
FALCON3_PRECISION_CONFIG=<config> TTNN_CONFIG_OVERRIDES='{"throw_exception_on_fallback": true}' python models/autoports/tiiuae_falcon3_7b_base/tests/full_model_evidence.py --model-dir models/autoports/tiiuae_falcon3_7b_base --reference results/reference/aime24_100.refpt --output <candidate>/full_model_evidence.json
python models/autoports/tiiuae_falcon3_7b_base/doc/datatype_sweep/generate_artifacts.py
```

Exact expanded commands are stored per row in both sweep result files. Final
review returned `clean-pass` after remediation and four fresh review passes;
the final report is `stage_review.md`. Stage implementation/evidence checkpoint:
`15048337ac4` (`Select Falcon3 full-model precision policy`). No push was made.
