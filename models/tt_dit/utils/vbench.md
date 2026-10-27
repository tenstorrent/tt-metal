# LTX CI quality gate

The Fast T2V CI test generates five 145-frame, 1920×1088 clips. After pytest closes
the TT devices, the same worker runs `vbench_bundle evaluate` with up to five CPU
processes, four Torch threads each. Generation and scoring share the existing
35-minute test timeout. The commands live in `tests/pipeline_reorg/models_e2e_tests.yaml`.

The evaluation policy splits metrics by their input needs:

| Metrics | Input |
| --- | --- |
| Subject consistency, background consistency, imaging quality | Original clips, with VBench's normal preprocessing |
| Motion smoothness, dynamic degree | 960×544 copies, all 145 frames, original frame rate |

The temporal copies use PyAV in-process, with area downsampling and lossless RGB
encoding; source frame timestamps are preserved and no OS command is launched. This reduces
sensitivity to small spatial motion artifacts; it is **not equivalent to evaluating
temporal quality at full resolution**. It preserves every seed, frame, metric and
the existing mean thresholds. Resize fails if the frame count or frame rate changes.
Missing seeds, worker failures, mismatched clip hashes, missing metrics and non-finite
scores all fail the gate.

Dynamic degree uses VBench 0.1.5's original RAFT model, frame sampling, flow threshold
and required positive-pair count. It stops once the remaining pairs cannot change
the binary result. Unlike the spatial reduction, this early exit preserves the
reference result exactly; exhaustive small cases test both passing and failing bounds.

On 2026-09-18, the five actual CI clips from run 35334111844 passed this complete
gate on g15blx02 in 343 seconds using the CI entry point and the in-process resize:

| Metric | Mean | Required |
| --- | ---: | ---: |
| Subject consistency | 0.9396 | 0.92 |
| Background consistency | 0.9590 | 0.93 |
| Motion smoothness | 0.9890 | 0.955 |
| Dynamic degree | 0.8 | 0.8 |
| Imaging quality | 0.6831 | 0.645 |

A separate saved passing clip also passed the reduced temporal policy (motion
smoothness 0.9911, dynamic degree 1.0). Freezing its frames failed dynamic degree
(0.0); alternating inverted frames failed motion smoothness (0.3110).

These measurements cover CPU scoring, not cold model startup. CI must also fit
weight conversion, kernel compilation, generation and CLIP inside the total budget.

`VBENCH_TEMPORAL_WIDTH=960` selects the CI policy when exporting a bundle.
The manifest records this setting and result files identify the exact manifest.
An exported width of `0`, or a local pipeline run without `VBENCH_EXPORT_DIR`,
uses the original full-resolution evaluation. Evaluator weights come from the
worker's existing `VBENCH_CACHE_DIR`, `TORCH_HOME` and CLIP caches.
