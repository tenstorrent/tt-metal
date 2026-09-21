# B compensation phase findings

The sampled private clone of the frozen v1 B winner is bitwise equal to both
the original and disabled wrapper. Capture `B-phase-profile-01` contains all
300 expected timestamp markers: ten steps for eight numerator batches and two
denominator batches on each of three RISCs. Decoded data are in
`B-phase-profile-01-decoded.json`; raw CSV is retained in the capture directory.

Whole-kernel MATH zone: baseline 449,729 cycles, disabled wrapper 449,816,
instrumented 452,328 (+0.578% relative to baseline). This is a small Q-repeat1,
K-chunks8 profiling case, not the uninstrumented throughput benchmark.

PACK RISC median control-issue intervals (cycles):

| Sample | Whole batch | SFPU issue | Subsequent SFPU drain / pack issue | Release |
|---|---:|---:|---:|---:|
| Numerator (8 batches) | 637 | 133 | 262 | 98 |
| Denominator (2 batches) | 780.5 | 162.5 | 403 | 65 |

These are **instruction-issue timestamps, not engine-retirement times**.
The middle interval includes backpressure and cannot be labeled pure PACK
cost. Approximately ten markers ×25 cycles also perturb each selected batch
substantially despite the small whole-kernel overhead. These measurements
locate possible scheduling boundaries; they do not establish original kernel
critical-path percentages or a quantitative speedup ceiling.

Cross-thread timestamps show UNPACK/MATH preparation overlapping the preceding
PACK batch. Sampled acquisition intervals are near the ~25-cycle marker floor.
Combined with the nearly neutral correction-cache result (337.116→336.758 ms,
0.106% in one paired screen), this motivates trying cross-half SFPU/PACK overlap
instead of assuming another copy/broadcast setup reduction will materially
help. A future focused measurement should use only two endpoint markers to
reduce local instrumentation distortion.
