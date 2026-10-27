# LTX CI quality gate

The Fast T2V CI test generates five 153-frame, 1920×1088 clips. After pytest closes
the TT devices, the same worker runs `vbench_bundle evaluate` (`models/tt_dit/utils/vbench_bundle.py`)
with up to five CPU processes, four Torch threads each. Generation and scoring share the existing
35-minute test timeout. The commands live in `tests/pipeline_reorg/models_e2e_tests.yaml`.

With `VBENCH_TEMPORAL_WIDTH=960` the test scores the clips with `vbench_cpu.score_clip_ci`:
appearance metrics on the original clips, motion smoothness and dynamic degree on 960-wide copies.
The policy and how the copies are made are documented in `models/tt_dit/utils/vbench_cpu.py`.
Missing seeds, worker failures, missing metrics and non-finite scores all fail the gate.

On 2026-09-18, the five actual CI clips (then 145 frames) from run 35334111844 passed this complete
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

Evaluator weights come from the worker's existing `VBENCH_CACHE_DIR`, `TORCH_HOME` and CLIP caches.
