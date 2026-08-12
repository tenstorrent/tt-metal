# AutoDebug: canonical BFP8/BF16 LM-head L1 overflow
## Status

Source-only diagnosis complete; no implementation files were edited and no TT
hardware was opened. The repo AutoDebug runner was invoked first as required,
but its nested Codex sandbox could not launch shell reads because `bubblewrap`
was unavailable. The findings below were then checked directly against the
existing logs, current source, and prior optimized-full-model artifacts.

## Direct observations

- Both BFP8 candidates fail with exactly `1778432 B > 1572864 B`; changing the
  LM-head fidelity from LoFi to HiFi2 does not change the allocation.
- BF16/HiFi4 fails with `2745088 B > 1572864 B`.
- The traceback is unambiguous: `generator.prefill_forward` calls
  `model.prefill_selected_hidden_logits`, which calls `LMHead1D.forward`, and
  the failing operation is its `ttnn.linear`. This is the terminal LM head,
  not a decoder prefill matmul.
- `_build_lm_head` constructs one 32768-column local-vocabulary weight split,
  uses the decoder's 32-core `residual_grid`, DRAM width-sharded weights, and a
  hard-coded `_dram_matmul_program_config(..., in0_block_w=3)`.
- Prior optimized-full-model evidence proves that this exact one-piece geometry
  was selected only for BFP4/LoFi weights and BFP8 output. For that policy,
  legal K block widths 1 and 3 both ran; width 3 was faster. That evidence does
  not establish L1 feasibility for BFP8 or BF16 weights.
- The prior adapted 8-core terminal layout is a different geometry and already
  overflowed at 2192192 B even with BFP4. The adapted 16-core layout ran but was
  slower than the selected BFP4 32-core path. Neither result rejects a
  precision-specific 32-core width-1 or smaller-column split.

## Headline finding 1: the BFP4-tuned LM-head program geometry is reused across weight dtypes

The selected precision policy changes `weight_groups.lm_head`, and LazyWeight
correctly materializes that dtype, but `_build_lm_head` does not adapt the
program configuration to it. It always uses 32768 local columns and
`in0_block_w=3`. Increasing the weight tile storage from BFP4 to BFP8/BF16
therefore increases the DRAM-sharded matmul's static circular buffers while the
per-core L1 budget and geometry remain fixed. The identical BFP8 allocation for
LoFi and HiFi2 also shows this is a dtype/geometry capacity failure, not a
compute-fidelity failure.

Prediction: keeping the same 32-core residual contract but using the already
legal `in0_block_w=1` will materially reduce K-block-related circular buffers
and is the smallest likely fix for BFP8. BF16 may still overflow because it is
1172224 bytes over budget, versus only 205568 bytes for BFP8.

Smallest verify/refute experiment:

1. Add a temporary/configurable LM-head `in0_block_w` override only; do not
   alter decoder geometry, weights, output dtype, or fidelity.
2. Construct one real-weight terminal LM-head call for BFP8/LoFi at widths 3
   (control) and 1, with strict fallback. Expected: width 3 reproduces 1778432;
   width 1 compiles and returns matching logits/tokens.
3. Repeat BF16/HiFi4 width 1. Record its exact allocation if it still fails.
4. Only after construction passes, run the full original readiness command;
   compare accuracy and traced teacher-forcing speed. The BFP4 sweep already
   shows width 1 is legal, but not that it fits or is competitive at other
   dtypes.

## Headline finding 2: the one-piece 32768-column split is an optimization, not a dtype-independent contract

`LM_HEAD_COLUMNS_PER_DEVICE = 32768` forces one local-vocabulary projection.
Earlier optimized-full-model evidence measured 8192, 16384, and 32768-column
forms with real BFP4 weights and selected 32768 for speed. `LMHead1D.forward`
already supports multiple weight/program-config splits and concatenates their
outputs. Thus reducing the column split is an existing semantic path, not a new
LM-head design. A smaller N split reduces per-core output work/buffering and is
the next focused adaptation if width 1 alone does not fit, especially for BF16.

Prediction: 16384 columns with `in0_block_w=1` should require less per-core L1
than 32768 columns; 8192 is the next bounded control. It will add launches and
concat overhead, so it must be ranked using traced full-model teacher forcing,
not assumed performant.

Smallest verify/refute experiment:

1. With the same real weights, residual grid, dtype, fidelity, output dtype,
   and strict-fallback setting, test this ordered matrix independently:
   `(32768,1)`, `(16384,1)`, `(8192,1)` for each still-failing dtype.
2. For every failure capture the exact circular-buffer byte count. For every
   pass compare local logits or generated tokens to the matching higher-
   precision/control path.
3. Run the original readiness command only for the smallest-launch-count
   passing geometry, then measure traced teacher forcing. Do not infer the best
   split from the old BFP4 timings.

## Refuted or lower-priority hypotheses

- **Decoder prefill geometry:** refuted for this symptom by the traceback; the
  decoder has completed and failure begins in terminal `LMHead1D.forward`.
- **LoFi-specific kernel allocation:** refuted as the primary cause because
  BFP8 LoFi and HiFi2 allocate the identical 1778432 bytes.
- **KV-cache, CCL, or activation dtype:** these policy fields may affect other
  paths, but the first failure is weight loading followed by LM-head linear and
  scales with LM-head weight precision. Hold them fixed during the focused
  experiment.
- **Immediately adopting the prior 16-core terminal reshard:** lower priority.
  It changes layout and was already slower for BFP4; first exhaust the legal
  32-core width and column-split adaptations.

## Recommended repair boundary

Expose LM-head column split and K block width as derived precision-policy/runtime
fields, or deterministically derive a legal geometry from LM-head weight dtype,
rather than weakening the selected dtype. Preserve the BFP4 default
`32768/in0_block_w=3`; use measured passing geometry for BFP8/BF16. Whichever
fields are selected must appear in runtime propagation evidence so later model
construction does not silently return to the hard-coded BFP4 geometry.

## Evidence inspected

- `doc/datatype_sweep/results/bfp8_lofi_bfp8_act_ccl_kv/{run_teacher_forcing,full_model_evidence}.log`
- `doc/datatype_sweep/results/bfp8_hifi2_bfp8_act_ccl_kv/{run_teacher_forcing,full_model_evidence}.log`
- `doc/datatype_sweep/results/bf16_hifi4/{run_teacher_forcing,full_model_evidence}.log`
- `tt/model.py` (`_build_lm_head`, `prefill_selected_hidden_logits`)
- `models/common/modules/lm_head/lm_head_1d.py` (`LMHead1D.forward`)
- `doc/optimized_full_model/results/candidates/lm_head_program_config_sweep.json`
- `doc/optimized_full_model/{README.md,work_log.md,perf_summary.json}`
- `doc/optimized_decoder` geometry evidence for comparison only
