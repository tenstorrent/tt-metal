# BFP8 full-model performance and GPQA, October 10

The matched native/shared-QK/native comparison completed with clean device
closure. Native controls remained within 3%. One TP4 replica, 64 layers, 128
generated tokens, one warmup and three measured fresh-prefill samples per cell:

| Context | Batch/TP4 | Native TSU | Shared-QK TSU | Shared-QK output TPS/TP4 |
| --- | ---: | ---: | ---: | ---: |
| 32K | 16 | 11.754 | 14.871 | 237.931 |
| 32K | 32 | 7.343 | 10.601 | 339.247 |
| 16K | 16 | 12.646 | 16.337 | 261.393 |
| 16K | 32 | 8.040 | 12.126 | 388.046 |

32K gains are 26.5% at B16 and 44.4% at B32. These are decode-only measurements;
eightfold scaling is a projection, not a physical Galaxy or mixed-serving
measurement. The B32 projection is 2,714 output TPS/Galaxy at32K, 3,104 at16K.
Neither direct tiled-input preparation nor the fused epilogue is in this run.

## Full GPQA

**178/198 = 89.90%**, all questions in the denominator, in **65m54s**. Five
responses reached the65,536-token output budget; all five were scored incorrect.
No model-context cutoff occurred. The independent saved-response audit matched
every raw answer/reasoning hash, finish reason and token count to the scored
receipt. Completed-response score is also178/198, meeting the177-answer gate.

Protocol: corrected `preserve_scientific_notation_v1`, temperature1, top-p0.95,
top-k20, seed42, thinking enabled, DP8 x16 users, model context262,144.
Checkpoint and dataset pins are in the saved protocol. This is the corrected
local harness, not the separate OpenBench protocol. Mean decode speed16.309TSU
and aggregate739.216outputTPS are workload-level eval metrics, not32K fixed
context measurements. The accepted previous native BFP8 score was176/198,
with no truncations, in50m49s; timings from those different generated answers
must not be treated as matched kernel comparisons.

See [audit](gpqa-audit.json), [qualification](gpqa-qualification.json),
[queue](perf-priority-v1/queue.json) and the unmodified sweep/comparison receipts.
Raw answer text stays on host disk; only hashes and public scoring receipts
were exported. `capture.json` records original file hashes.

## Bandwidth interpretation

At32K, the existing BFP8 useful-traffic model counts12.884GB/chip/step atB16
and18.655GB atB32: padded weights once, KV once and FP32 state read/write.
The measured times imply191.6/197.8GB/s per chip, or37.4%/38.6% of assumed
512GB/s peak. These are useful-byte roofline fractions, not DRAM counters.
Compute, extra traffic, collectives and synchronization are excluded from the
ideal39.74/27.45TSU ceilings. A twofold throughput target means29.74/21.20TSU
and33.62/47.16ms steps, approximately74.8%/77.3% of this simplified ceiling.

## Separate container result

The earlier native-BFP8 image launched all eight TP4 workers and passed API
and tool-call smoke checks. Its OpenBench launch failed before requests:
the staged filename `openbench.py` shadowed the installed `openbench` package.
Keep the launcher's original `run_openbench_gpqa.py` name on retry and verify
imports before allocating hardware. This is not a failed model evaluation.
The owned container was removed; its dirty marker was subsequently recovered
by the safe runner before the passing direct-preparation test. Container eval,
Shield Galaxy CI and release promotion remain separate gates.
