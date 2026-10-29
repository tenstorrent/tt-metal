# Phase review, round 2 (composite-bringup), 2026-10-29

Verdict: clean-pass

Both round-1 required items verified resolved with hardware evidence and re-derivable numbers: clock_analysis.py reproduced byte-for-byte on the four full CSVs (1187/1171/1146/1130 MHz, idle control 1349-1350, in-step idle 0.2-1.1 %, inter-step 1.5-2.2 ms); the tt-smi trace's 15 in-window samples read 1125-1200 MHz (mean 1151), never 1350; compare_loss.py reproduced exactly (max 0.0754, PASS, determinism True); the bf16-params control reproduces step 6 and is a faithful model of the metal AdamW kernel; harness at HEAD 11 passed, llama tests 131 passed; no stale "288 ms / 14 % host gap" text remains.

Other concerns (all text, fixed in the closing commit): circular 8-layer validation sentence (restated with independent clocks: 445.1 / 437.1 ms vs 443.8 / 434.0; 40-layer wall range 1933-1985 ms); tolerance derivation leaned on the quantised 0.019 (restated as the 1 % bar on its own, sign clause reported not asserted); residual +0.015 offset attributed in prose (now a transient of steps 2-5 with a recorded hypothesis, binomial p 0.11); stale "comparison bounded by bf16" sentence; "1-2 % noise" was a 4.5 % thermal clock effect; tt-smi sample accounting (15 in-window, 1125-1200 MHz); plan's "recorded in the README" for power-vs-thermal (now "not determined"); "133 passed" -> 131; the 1620 ms target includes 58 ms of planned dispatch, so the kernel budget is 1562 ms and the baseline is 6 % above it.

Hard-check gaps: power-vs-thermal cause of the clock (hardware probe, outside the phase); the loss comparison is a script, not a tests/python test; the steps 2-5 residual mechanism is a hypothesis.

Residual risk: the delivered clock varies 1130-1160 MHz with thermal state, so every loop MFU comparison carries its own clock; bf16 adapter/moment storage loses up to a third of each update (stochastic rounding / AdamWFullPrecision candidate before long runs); all 40-layer figures are projections (1933-1985 ms at the delivered clock).
