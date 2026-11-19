# Production substitution in the unchanged legacy probe

Prepared source-only; no device run by the audit agent. New wrapper:
`tests/multichip_conv_gates_production_legacy_probe.py`.
New root-runner queue: `doc/optimized_multichip_decoder/conv_gates_production_legacy_jobs.json`.

```sh
TT_METAL_WATCHER=10 TT_METAL_WATCHER_NOINLINE=1 TT_METAL_WATCHER_APPEND=1 \
python -m models.autoports.ornith_ai_ornith_1_5_9b.tests.multichip_conv_gates_production_legacy_probe \
  --output models/autoports/ornith_ai_ornith_1_5_9b/doc/optimized_multichip_decoder/conv_gates_production_legacy_direct_watcher.json
```

Run this command only through the root's serialized hardware runner. Keep the default batches, waves and eight steps: narrowing to B13 alone changes the RNG state and does not reproduce the original fixture.

The wrapper imports the unchanged legacy probe, replaces only its process-local `fused` module, and invokes its original `main`. It uses current production `tt.tiled_conv` with the supplied independent history and then the exact separate TTNN beta/a/g assignments from the legacy control. CPU AST comparison confirms those three assignments are identical. Reshaping the resulting gates from `[B,1,8]` to `[1,1,B,8]` supplies the original packed-gate recurrence's one-tile input contract; no gate values are recomputed. The original recurrence, seed7903, random draws, independent state/history, all per-head/global .995 checks, bitwise history checks, saved failure tensors and raised assertions remain intact. The wrapper adds production-helper source hashing without changing original results or exception status.

Expected discriminator: if this production substitution also fails the legacy per-head output check around B13/wave104/step5, while beta/g match their identical separate-TTNN references, the original mismatch is reproducible without fused gate arithmetic. This establishes the legacy convolution-reference mismatch against accepted production rather than dismissing a candidate-specific failure. The exact PCC need not equal .9882502 because this control removes the old softplus-polynomial difference. A pass would refute the claim that production convolution alone necessarily reproduces that failure; retain it and investigate further. Never turn an expected failure into a pass or relax .995.

Existing hardware evidence was read, not rerun: `conv_gates_precise_matched_direct_watcher_wide_modes.json` is complete with704records; fused convolution versus production and same-gate state/output are bitwise equal in every record. Main output versus production separate-gate control has minimum per-head PCC .999977946. The original separate-TTNN-convolution failure remains preserved. This new probe adds the missing direct production-versus-legacy-output comparison; the earlier matched probe compared production and legacy convolution but did not run a separate legacy-convolution recurrent output for that comparison.

CPU validation: `py_compile` passed; gate-assignment AST identity and source hashes are recorded in `production_legacy_validation.json`. No existing test, kernel, helper or production file was edited.
