# AutoFix Report

## Starting Evidence

- Fresh diagnosis: `AUTODEBUG.md`.
- Original failing commands/logs: all-projection BFP8 HiFi2 and LoFi traced
  teacher-forcing runs; preserved as `logs/*_pre_autofix.log`.
- Both failed in the first layer MLP-down projection with a 14,656-byte static
  circular-buffer/L1 overlap.

## Hypothesis Experiment

- Hypothesis: BFP8 MLP-down retained the BFP4-tuned 17-tile K block because two
  multichip call sites ignored `OptimizationPolicy.mlp_down_in0_block_w`.
- Prediction: consuming the field and setting width 1 for BFP8 eliminates the
  compile clash without changing the BFP4 selected path.
- Fix: both MLP-down paths now consume the policy field; all-projection BFP8
  candidates set width 1, while the BFP4 selected config retains width 17.
- Verification:
  - BFP8+HiFi2: 97/100 top-1, 100/100 top-5/top-100, 6.26 traced t/s/u.
  - BFP8+LoFi: 98/100 top-1, 100/100 top-5/top-100, 6.64 traced t/s/u.
- Verdict: verified and fixed. Both original commands pass; neither candidate
  beats selected BFP4+LoFi at 7.00 traced t/s/u.

## Final Status

Fixed with full-model traced accuracy/performance evidence. No speculative edit
was retained.

## Review remediation: BFP8 residual boundary

- Hypothesis: reduced BFP8 activation/residual plus BFP8 CCL is a legal
  candidate once its dtype is propagated to embedding and TP4 projection/CCL
  outputs.
- First experiment: the full-model traced command reached
  `nlp_create_qkv_heads_decode` and failed its exact BF16/FP32 input contract.
- Focused fix: locally typecast packed QKV to BF16 immediately before that op;
  keep the residual/projection and CCL boundaries BFP8.
- Verification: original full-model command passes at 97/100 top-1,
  100/100 top-5/top-100, TTFT 5170.21 ms, and 5.97 traced t/s/u.
- Verdict: legal but slower; selected BF16 residual remains the winner.
  The pre-fix log is preserved as
  `logs/selected_bfp8_activation_ccl_teacher_forcing_pre_autofix.log`.
