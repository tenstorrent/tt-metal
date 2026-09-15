# Full-HiFi4 accuracy investigation

Completed 2026-09-11. See [REPORT.md](REPORT.md) for results, limitations,
reproduction instructions, and verified source restoration. The modes below
are preserved in diagnostic patches, not active in the retained build.

Retained operator snapshot is backed up at `/tmp/sdpa-accurate-base.aAL0RT`
locally and its four source hashes are recorded by qualification-v1.
No qualification thresholds are changed by this investigation.

All diagnostic modes use original BF16 Q, HiFi4 QK/PV and matched denominator,
FP32 destination/state, Q/K chunks 128/1024, and unchanged input buffering.
`TT_SDPA_ACCURACY_DIAG` selects a compile-time diagnostic in the temporary host
factory. Each mode is run in a fresh Python process; it is not a production API.

| Mode | Scores into subtraction | Logit exp |
|---:|---|---|
| 0 | Existing FPU operand path | Retained biased grid/cubic, control |
| 1 | Existing FPU operand path | Refit same grid/cubic without six-bit bias |
| 2 | Existing FPU operand path | Accurate SFPU exp |
| 3 | Full-FP32 alias and SFPU subtraction | Accurate SFPU exp |
| 4 | Full-FP32 alias and SFPU subtraction | Unbiased grid/cubic |

The FP32 alias shares existing score storage, synchronizes its read pointer
from the original score CB, and adds no Q/K/V movement or score-buffer copy.
The first diagnostic uses a conservative one-score-tile-at-a-time schedule;
its throughput is not assumed representative of an optimized implementation.

First use the six failure-directed heads from stress-analysis, preserving
input hashes and original FP64 reference positions. Then expand validation to
all heads/seeds of remaining supported qualification cases, fresh holdouts,
and structural probes. Measure performance only after numerical attribution.
Unsupported streaming inputs must not silently fall back.
