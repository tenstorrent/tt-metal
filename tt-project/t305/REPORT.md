# #305: main vs ttp/ltx23-main-pr, LTX-2.3 8+3 standard e2e (blx01 4x8)

Command (both arms, unmodified test, bf16, seed 10, 8+3):
`pytest models/tt_dit/tests/models/ltx/test_pipeline_ltx_distilled.py::test_pipeline_distilled -k 4x8sp1tp0nl2_ring_is_fsdp0`
via `bash /var/tmp/fasth3/t305/run305.sh <arm>` (= t301/run301.sh + disk caps), env t301/env.yaml, -t 570.
main = origin/main 80b1cd689d0, PR = ttp/ltx23-main-pr df9e5ecaac6. Warm gen #2 table quoted.

| job | arm | AICLK | Enc | S1 | Up | S2 | VAE | Audio | Total |
|---|---|---|---|---|---|---|---|---|---|
| 210 (#301) | main | 1350 | 0.45 | 2.11 | 0.11 | 2.55 | 0.82 | 0.37 | 6.41 |
| 214 | main | 1350 | 0.19 | 2.14 | 0.11 | 2.54 | 0.82 | 0.39 | 6.19 |
| 213 (#301) | PR | 1350 | 0.47 | 2.08 | 0.11 | 2.40 | 0.82 | 0.37 | 6.24 |
| 215 | PR | — | dropped (chips 16-23 left PCIe, 2026-10-09 02:45:36 UTC) |
| 349 | main | 900 clamp | 0.25 | 2.93 | 0.14 | 3.37 | 0.93 | 0.47 | 8.09 |
| 242 | PR | 900 clamp | 0.26 | 2.83 | 0.14 | 3.20 | 0.92 | 0.47 | 7.83 |

Verbatim gen #2 table, job 349 (main, clamped):
```
│ Encoder                                                         │ 0.25 s │
│ Stage 1 denoise                                                 │ 2.93 s │
│ Latent upsample                                                 │ 0.14 s │
│ Stage 2 denoise                                                 │ 3.37 s │
│ VAE decode                                                      │ 0.93 s │
│ Audio decode                                                    │ 0.47 s │
├─────────────────────────────────────────────────────────────────┼────────┤
│ Total                                                           │ 8.09 s │
```
Job 242 (PR, clamped): Encoder 0.26 / S1 2.83 / Up 0.14 / S2 3.20 / VAE 0.92 / Audio 0.47 / Total 7.83 s.
Full logs: results/*.log.gz (213/210 in tt-project/t301).

## Findings
- Gain the test measures: about -0.2 s. Unclamped, excluding the noisy Encoder row: main 5.96/6.00 vs PR 5.77
  (-0.19..-0.23 s). Clamped same-condition pair: -0.26 s (scales with the 1350/900 clock ratio, ~-0.17 s at full clock).
  All of it is in the DiT: Stage 2 -0.15..-0.17 s, Stage 1 -0.03..-0.10 s. VAE, upsample, audio unchanged.
- Honest PR claim: "~0.2 s (~3%) lower Total in test_pipeline_distilled 8+3 on 4x8 (Stage 1 + Stage 2 denoise)".
  Do not claim the ~0.4 s.
- Where the missing ~0.2 s went:
  - Export gains (5565232f12c AAC overlap, df9e5ecaac6 x264 ultrafast) run after `last_timings` is set
    (pipeline_ltx_distilled.py ~line 856), so the standard table never sees them. Real for users, not in this number.
  - mel-VAE audio trace (05401286709) does run (traced=True, one more trace captured) but saves ~2 ms: main's eager
    audio decode is already 0.37 s; the ltx-rt baseline it was measured against (~0.88 s) does not apply on main.
    Dead in this path: propose dropping it from the PR.
  - Seeded-noise prefetch (bfc93527e92): keys match by code reading, so it is used; it is inside the S1/S2 gain.
