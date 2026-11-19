# Selected-step real-input GDN diagnostic

Added `tests/run_decode_sequence_diagnostic.py`. This is a diagnostic test helper only: no decoder edits, no gate change, no hardware execution by the author. It preserves the seed, HF constructor ordering, 128-token prefix, independently sampled 128-token stream, warmup/capture/restoration, and traced output comparisons from `run_decode_sequence.py`.

Use the **exact decoder candidate and policy environment of the failing run**, replacing only the runner. For example, with that environment already exported:

```bash
QWEN_DECODER=optimized python_env/bin/python -m models.autoports.qwen_qwen3_6_27b.tests.run_decode_sequence_diagnostic \
  --layer 0 --batch 32 --steps 128 \
  --target 19:20 --target 58:30 --target 99:27 \
  --dump-intermediates models/autoports/qwen_qwen3_6_27b/doc/optimized_decoder/sequence_capture \
  --output models/autoports/qwen_qwen3_6_27b/doc/optimized_decoder/sequence_capture.json
```

Repeat with distinct artifact directories and the measured native-chunk, `delta_direct`, or `native_recurrent` policies. The script respects `QWEN_OPT_CANDIDATE` and `QWEN_OPT_POLICY`; it does not override them. `--target STEP:USER` is repeatable, zero-based, and defaults to the three coordinates above. To inspect a different earliest failure, use its coordinate from the preceding sequence JSON. Omitting `--dump-intermediates` still writes scalar/per-head diagnostics. The .995 prefix and minimum per-user gates remain; the runner writes evidence **before** raising on a failed gate. A diagnostic failure must not be reported as a pass.

## What the hooks observe

At each selected step, the script saves the full TT convolution/recurrent buffers before and after the **ordinary uninstrumented trace replay**. It restores the pre-step buffers and runs that same token eagerly with temporary hooks. It then restores the traced post-step buffers in `finally`, so the sequence advances once. HF advances exactly once, with hooks on its actual recurrence and modules. Hooks are never installed during trace capture/replay. All host conversions are diagnostic-only; this runner reports no performance claims.

The output JSON contains the complete traced PCC grid, prefix state comparison, eager-versus-traced output and both state comparisons, and comparisons at these boundaries:

- Input RMSNorm, raw qkv/a/b projections, convolution history, post-convolution q/k/v, g, beta and z.
- Recurrent input state, core output, updated state, and persistent cached state.
- GDN RMSNorm input, ungated weighted norm output, gated output, attention projection output, post-attention RMSNorm, and final decoder output.
- Per-user/per-head core RMS and core/gated error statistics, exposing whether a small readout is amplified by normalization.

TT q/k are expanded to the 48 value heads **only on CPU for comparison**. Each `.pt` artifact stores selected-user HF/TT tensors, exact token input, traced output, HF norm weights, source provenance, and CPU attribution intermediates. Actual native internal normalized q/k, prediction and delta are not exposed by its API; they are explicitly CPU oracle results, not claimed device captures. The HF ungated norm intermediate is reconstructed with the installed implementation's BF16 multiplication boundary and FP32 rsqrt. TT RMSNorm's native return is also retained when a subsequent cast may narrow it.

Five CPU direct-recurrence combinations separate factors without changing the acceptance reference: HF inputs/HF state/HF normalization; TT inputs/TT state/HF normalization; TT inputs/TT state/FP32 normalization; HF inputs/TT state/HF normalization; and TT inputs/HF state/HF normalization. They store normalized scaled q, normalized k, exp(g), prediction, delta, updated state and pre-pack core. They are IEEE-FP32 controls, not emulations of TF32/native matrix arithmetic. Summaries compare each oracle to HF core, TT core and TT updated state.

## Falsifiable interpretation

1. Check eager-versus-trace exactness first. A disagreement makes eager intermediates insufficient to explain the traced failure. The helper restores public state but cannot prove hidden scratch has no persistent state; compare its traced PCC grid to the original uninstrumented run before drawing conclusions.
2. Input-norm or projection differences preceding recurrence identify an upstream discrepancy. Compare a/b as well as g/beta; weight precision alone does not control norm rounding.
3. Compare each native/direct output to the CPU oracle on **its own exact inputs and prior state**. A close match there, combined with poor HF core agreement, directs attention to inherited state/upstream differences. A poor own-input match implicates recurrence arithmetic, normalization policy or its implementation.
4. Compare crossed-input/state oracle results to discriminate pre-existing state error from this token's upstream error. These are one-step counterfactuals, not replacement sequence accuracy measurements.
5. Compare core, ungated norm, gated output and attention projection. A large relative/angular change after norm must coincide with the measured head readout magnitudes before calling it cancellation amplification.

## Validation and limitations

`python3 -m py_compile .../tests/run_decode_sequence_diagnostic.py` passed. A CPU-only AST extraction of `cpu_recurrence` (no TTNN import) matched the installed HF `torch_recurrent_gated_delta_rule` exactly for synthetic BF16 q/k/v/beta/state, FP32 g, and S=1: BF16 output and FP32 updated state were bit-equal. This validates the attribution oracle's operation order, not the unexecuted device instrumentation.

No hardware was opened. Device execution remains unverified. The helper deliberately supports the optimized/fused `_delta_chunks` topology and GDN layers only; it fails explicitly if required hooks are absent. Temporary full-batch host snapshots cost about 200 MiB for two recurrent states at B32, plus comparison temporaries. Selected-user tensor artifacts are much smaller. Run serially under the caller's existing device-usage reservation.
