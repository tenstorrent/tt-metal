# Tafsut optimization log

Target: one traced forecast at the `paper` shape (1024 series x context 2048 -> 64), `accuracy` precision, device
time from `tests/perf/test_forward_perf.py` (median of 5 runs, each after a 0.3 s idle). Guards: `long_context`
(32 x 32768 -> 64) and `long_horizon` (512 x 2048 -> 2048, two forwards) at accuracy, and `paper` at `performance`.

A change was kept at >= 2 % on the target's device time, with no guard more than 3 % slower and every floor in
`tests/perf/floors.json` holding on the full bench and PCC suite. Job numbers are the runs on one Blackhole p150a.

| id | round | change | status | evidence |
|---|---|---|---|---|
| R0 | 0 | baseline: the Chronos-2 structure (fused QKV, masked SDPA, traced forward, read-out cut in the last layer) | kept | paper 1697.5 ms accuracy, 1654.2 ms performance (job 1061) |
| R1-1 | 1 | head split + RoPE fused in one model-local op (`ops/qkv_heads_rope`, ported from chronos_forecast, fp32 DST); matmul RoPE where the cos/sin table does not fit in L1 | kept | 4-layer screen -43 % (jobs 1186, 1187) |
| R1-2 | 1 | 2D multicast matmul program configs over the whole grid, per role | kept | -58 % on top of R1-1 in the screen; paper accuracy 1699.2 -> 408.6 ms (job 1220) |
| R1-3 | 1 | ReLU fused in the ff-up and head-hidden program configs | kept (with R1-2) | job 1220 |
| R1-4 | 1 | residual add fused into the next norm (`ops/add_rms_norm`) with the RMSNorm gamma folded into the next weights | dropped | paper accuracy -3.6 % (job 1298), but `median_err` 0.0384 -> 0.0664 and 17 other floor failures together with R1-7a (job 1333); still available as `TAFSUT_ADD_NORM=1` |
| R1-5 | 1 | SDPA writes the heads concatenated (`output_concat_heads`) | kept | paper accuracy 408.5 -> 399.7 ms (-2.2 %, job 1297); gated metrics bit-identical |
| R1-6 | 1 | `long_context` SDPA chunk micro-benchmark, 1088/32 against 1152/64 | not run | deferred to round 2 |
| R1-7a | 1 | fp32 dest accumulation off in the performance preset's linears | dropped | -2.2 % (job 1295), but `gift_ragged`/performance `abs_err_max` 0.05 -> 0.116 and `long_horizon`/performance floors fail (job 1333); still available as `TAFSUT_PERF_BF16_DEST=1` |
| R1-7b | 1 | RMSNorm gamma fold; `exp_approx_mode` in SDPA | dropped with R1-4 (fold); `exp_approx_mode` not tried | job 1333 |
| R1-8 | 1 | QKV HiFi4 -> HiFi2 in the performance preset | no change | the performance preset already runs QKV at LoFi |

Round 1 end: paper accuracy 1697.5 -> 399.8 ms (-76.5 %, 4.25x; bench job 1334, PCC suite job 1335).

Round 2 was not started. Its first candidates are R1-6 and keeping the norm gamma in the `add_rms_norm` kernel
(instead of folding it into the next weights), followed by a fresh profile of the round-1 code.
