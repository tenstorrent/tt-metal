# Real-weight GDN epilogue result

Completed October 9 at 23:09 UTC with a passing hardware test and clean device
closure. All four TP4 ranks produced bit-identical projected outputs and states
for native/fused/native controls. The test uses real layer-0 BFP8 weights,
64 FP32-reference updates, persistent output buffers and traced timing.

| Batch | Native us | Fused us | Block throughput ratio | Control drift |
| --- | --- | --- | --- | --- |
| 32 | 1016.36 | 951.70 | 1.0679 | 0.0605% |
| 16 | 710.69 | 685.47 | 1.0368 | 0.0289% |
| 8, native fallback | 559.16 | 559.61 | 0.9992 | 0.0045% |
| 1, native fallback | 364.02 | 364.18 | 0.9996 | 0.1362% |

The measured scope is projection, convolution, shared-Q/K recurrence, epilogue
and output projection; MLP is excluded. B16/B32 select the fused epilogue.
B1/B8 exercise unchanged native fallbacks. All timing comparisons pass the
3% bracket-drift limit. Each arm has five samples of 30 trace replays.

Multiplying the block savings by 48 GDN layers projects 3.103 ms per model step
at B32 and 1.211 ms at B16. Neither full-model speedup nor model-quality
qualification is established by this layer test. The running full-model queue
still tests shared-Q/K with the native epilogue from its frozen source.

Raw cases, reference checks, timing samples, hashes, final queue/JUnit and unit
status are retained here. The exact source and launch are in
`../gdn-epilogue-layer-launch-v1`.
