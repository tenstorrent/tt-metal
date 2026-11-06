# GPQA smoke rung valid

Completed IDs152 and189, both correct: **2/2 (100%;95% Wilson interval 34.24%–100%)**. Dataset198, completed2; lm-eval's effective198 metadata must not be reported as completed coverage. The two responses finish with boxed C and boxed A matching the targets. The task's boxed-choice grade agrees with manual final-answer inspection.

Chat-only lm-eval protocol supplies no tools or task executor. No response URLs, timeout/error, fetched-fix evidence or incomplete final answer observed. Raw finish_reason and usage were not persisted and remain unknown. Native thought text was included in content on this pre-parser run, but the final boxed answers are valid. The interval is too wide to establish parity against published84.3% or GPU83.33% full-suite references.

Evaluation wall **163.384s** at C2/B8. A ten-task nightly at the same concurrency projects **13.62minutes** of evaluation by five measured waves, before build/startup/queue and tail margin. Do not treat this tiny-sample extrapolation as a release projection with measured tail coverage.

Release settings confirmed in results:14400s timeout,126976output cap, temperature1/top_k20/top_p0.95, seed42 and pinned revision. Coordinator verified full60 autoport registration,1x4mesh,262144context andB8 in full job log. No device work was performed for this audit. See audit.json for compact evidence references and unknown measurements.
