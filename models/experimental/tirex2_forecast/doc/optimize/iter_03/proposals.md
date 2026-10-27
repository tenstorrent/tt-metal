# Round 3 proposals (ranked)

| id | candidate | expected | risk | measure |
|---|---|---|---|---|
| R3-1 | Step kernel on paper: init state (y, c, n, m) in ops/slstm_step (kernel session; requested in KERNEL.md), then the worker drops the loop on paper | -230 ms paper (-25 %) | medium (kernel session's work) | b2b paper both precisions, check_gates, suite |
| R3-2 | mLSTM chunkwise: fused activations in its binaries (as R2-2), no addcmul | -25..-40 paper, more on mv | low | slstm/mlstm module PCC, b2b |
| R3-3 | mLSTM TM cuts (R1-4 rest + R2-6): bmm transpose_b for kT, one stacked gate linear, fewer slices/concats | -25..-50 ms | low-medium | b2b + suite |
| R3-4 | Variate: subtract/keep/add as fewer ops | -40..-60 ms mv | low | mv b2b |
| R3-5 | sLSTM ys written in place (no 93-way concat) on the loop path | -20..-30 ms paper (until R3-1) | low | b2b |
| R3-6 | Host R1-7 (carried) | wall only | low | bench wall |

Order: R3-2, R3-3, R3-4, R3-1 as soon as KERNEL.md has the init support; R3-5 only if R3-1 is not coming.
