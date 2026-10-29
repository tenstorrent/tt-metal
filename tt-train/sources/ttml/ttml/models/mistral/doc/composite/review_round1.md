# Phase review, round 1 (composite-bringup), 2026-10-29

Verdict: more-work-needed

## Required Work
- P1: The 14.8 % "host/dispatch gap" in perf_analysis.md is a clock artifact: the device is ~100 % busy and runs at 1.13-1.25 GHz under load while Tracy converts cycles at 1350 MHz (`CHIP_FREQ[MHz]: 1350` in every profile_log_device.csv; idle-dominated step 1 gives exactly 1.35 GHz; tt-smi 1162 MHz at 168 W). Required: measure AICLK under load with tt-smi at 1 s over a >= 60-step run; re-express device time at the measured clock and redo the validation (wall = device at actual clock + inter-step sync + op-to-op idle); rewrite the class table, dispatch row and next items; state the clock every efficiency is quoted at; add a plan §3 calibration with the delivered clock and the target at it.
- P2: Loss-trajectory tolerance is post hoc (0.25 set to contain the step-6 outlier of 3.7 bf16 ulps) and the anomaly is dismissed by prose. Hypothesis from the artifacts: bf16 adapter/moment storage without stochastic rounding loses up to ~30 % of each ~1e-4 Adam update. Required: state the tolerance and derivation before the rerun; run torch_reference.py with A, B (and m, v) rounded to bf16 after each step as the control; report the ttml loss unquantised (fp32 on the host); close the ledger entry with numbers.

## Other Concerns
- "peaks at 17.4 GiB" is an after-forward read, not the peak (true peak ~1 GiB higher; 16-layer conclusion unchanged).
- Harness logs predate the implementation commits (loader fix landed between runs); rerun test_mistral.py at HEAD; no log for the claimed "133 passed" llama tests.
- `tracy_analysis.txt` was produced with `--top 45`, not the default; record it.
- BinaryNg rows in the Tracy CSV carry a stale [1024,6144] shape; the glue row merges five real sizes; recover shapes from the op sequence before costing.
- Attention-path gradient bars (0.98 / 8e-2) are relaxed but measured values sit 3-5x inside; block-level adapter gradients 2-3x worse than single-linear ones, no bias visible.
- Watcher log copy failed inside the chain (wrong path) but was copied later and is consistent (113211 lines, 0 errors).
- The loss comparison is a script + text, not a test under tests/python (Hard-Check Gap).

## Hard-Check Gaps
- AICLK under load needs the hardware trace; harness rerun at HEAD; a loss comparison that asserts its tolerance.

## Anomaly Ledger (reviewer)
- wall - device = 14-15 % at every depth: clock conversion + DVFS; more-work-needed (P1).
- step-6 loss diff -0.234 (3.7 ulps), steps 2-5 at +1.3-1.6 ulps: bf16 parameter storage suspected; no control run; more-work-needed (P2).
- tt-smi 1162 MHz / 168 W sample read as "1350 MHz under load": folded into P1.
- gate_up forward matmul 44 % vs dX 68 %: controlled, loop item.
- warm-chip rerun 453 vs 434 ms: controlled pending the clock measurement.
- BinaryNg stale shapes: controlled.

## Scope Inspected
Skills: stage-review, composite-bringup, optimization-loop measurement protocol. Artefacts: plan.md, perf_analysis.md, composite README and work log, every file under composite/evidence including the full Tracy CSVs and profile_log_device headers. Code: models/mistral/*, models/llama/*, ops/{linear_op,binary_ops,multi_head_utils}.cpp, modules/lora.py, examples/mistral_train/*, tests/python/test_mistral.py, optimizers/adamw.hpp, nanobind, tt_metal profiler sources. Commands: git log/show; analyze_tracy.py re-run (identical); python over the loss JSONs, npz, harness logs; device-cycle vs host-timestamp analysis over the full CSVs; no devices opened.

Contract check: implementation, freeze-map gating, layout contract, harness coverage (every module and layer kind, every gradient, fp64 on bf16-rounded inputs, frozen params gradient-free), loader, torch reference math, determinism, Tracy at four doubling depths with residuals <= 0.6 %, memory calibration, labelling "N of 40 layers": all met. The plan's single-chip substitutes (a)-(d) are met and honestly labelled.
