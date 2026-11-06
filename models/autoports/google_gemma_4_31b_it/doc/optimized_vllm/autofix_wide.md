# Wide batched prefill repair — diagnosis verified

Contract: batched real-weight prefill must preserve solo logits and valid KV data,
including arbitrary logical lengths and subsequent traced decode. The default
candidate remains capped at 1024 physical rows; no wide candidate is enabled.

Observation: B8 x 128 passes exactness; B32 x 128 with a probe-only 4096-row
limit changes first-layer K/V on all four chips and logits (max absolute
logit difference 9.15625). AutoDebug identifies a normalization dispatch change
at 1024 rows; causality was initially unverified; the following device experiments resolve it.

Prioritized hypotheses and discriminating checks:
1. Hidden RMSNorm reduction partition and reciprocal precision change beyond
   1024 rows. Compare identical embeddings through wide generic normalization,
   bounded sharded normalization, and solo normalization; then test bounded
   normalization with wide projections. Batched-prefill agent owns a probe only.
2. Wide DRAM activation or projection layout changes independently cause first
   K/V divergence. Diagnostics agent independently inspects these paths; isolate
   projections if normalization does not account for the first difference.
3. Retain the proven 1024-row grouping if wider projection/norm cannot preserve
   correctness or its added layout costs lose in real serving benchmarks.

No production normalization change has been made. Device experiments are
serialized behind the active compute-watcher acceptance matrix. No profiler is
used. Validation results and final disposition will be appended here.

Focused probe `norm_discriminator_b32.json` completed exit0. B32x128 real
embeddings are exactly equal to solo on all4 chips. For identical embedding
inputs, all four named generic-wide norms differ from solo, while bounded
sharded norms are bit-equal (zero differing elements) on every chip. This proves
an early numerical dispatch difference, not yet the sole cause of final logits.
The experimental bounded-norm prefill probe was run without production
normalization changes.

The experimental bounded-norm reduced prefill passes all32 rows of full-vocab
logits and valid K/V on layers0/5, all4 chips (`probe_b32_bounded_norm.json`).
This restores the previously failing wide B32x128 probe, supporting the hidden
normalization dispatch as the causal difference. No production wide path is
enabled. ABBA host-wall samples (input refresh through terminal/slicing; no
sampling) overlap: four B8 groups median29.046ms versus wide bounded-norm28.506ms.
These are reduced diagnostics, never full-model/device/serving performance.
The simpler <=1024-row grouping remains the primary serving candidate; the
wider option has no established serving gain and needs broader validation if
pursued. AutoFix has a verified repair experiment and a safe production bound,
not an unresolved shipped numerical divergence.

Disposition: retain the production1024-row bound. The wider experiment is repaired and understood, but is not selected. The real vLLM bounded candidate passes the required cold-burst gates; wider timing samples overlap the bounded control. No runtime numerical defect from the wide experiment is shipped.
