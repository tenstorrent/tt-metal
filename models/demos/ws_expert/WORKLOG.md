# ws_expert worklog

Weight-streaming routed expert, one expert's gate/up -> SwiGLU -> down on one Blackhole chip (generic_op), the model
of the "MoE Weight Streaming" artifact: H 3840, I 640 (20 pairs, KTD 20), bfp4 weights, LoFi, fp32 DEST. Average
M at 1M tokens = 586 rows (R 19 row tiles). Stage 1: x and weights resident in L1, h exchanged on chip.
Utilization: tile = tile products x 16 cycles / (cores x cycles) (padded rows); useful = at the real M.

## Single-core matmul rates (tests/test_mm_bench.py, resident, fp32 DEST, LoFi; peak 16 cyc / tile product)
| block | K | cyc / tile | % LoFi |
|---|---|---|---|
| 1x4 + SwiGLU (gate/up) | 120 | 17.87 | 89.5 |
| 1x4 plain | 120 | 17.28 | 92.6 |
| 2x2 / 1x2 + SwiGLU | 120 | 18.11 / 18.14 | 88.4 / 88.2 |
| 1x4 / 4x1 / 1x3 / 2x2 / 1x2 (down) | 20 | 17.41 / 17.42 / 17.55 / 17.67 / 17.85 | 91.9 - 89.6 |
| 1x1 | 20 | 27.8 | 57.6 |
| full-sync 2x4 + SwiGLU / 1x8 | 120 / 20 | 24.8 / 18.6 | 64.6 / 86.1 |
x bf16 vs bfp8: no difference. Full-sync DST loses the math / pack overlap: fp32 half-sync (4 tiles) 1x4 it is.

## Scheme B: dedicated gate/up (8 x 10 grid, stream-K rows) + 40 down cores (3 columns each, all rows) - dropped
- 120 cores. Correct (PCC 0.99998) but 509K cycles / expert (3.5%).
- Bugs on the way: a down reader credited row q + 1 only after row q's data arrived (one row in flight); the CONS
  multicast rectangle spanned the NoC x 8-9 gap (logical columns 0..7 = NoC 1..7, 10).
- Root cause: every h tile multicast to all 40 down cores. tests/test_mcast_bench.py on this box: a 4 KB multicast
  costs ~400-500 cycles whatever the sender count, aggregate into one rectangle 10-25 GB/s; 760 KB of h per expert to
  every down core cannot fit ~15 us.

## Scheme C: column clusters (ws_cluster.py, kernels wc_*.cpp) - current
- 9 clusters x 10 cores (one NoC column each) = 90 cores. Core j owns gate/up pairs {2j, 2j+1} (1x4 + SwiGLU, K 120)
  and down columns [12j, 12j + 12) (3 x 1x4, K 20). h stays in its column: each core multicasts its 2 h tiles of a
  row to the column (linked payload + flag, the mcast helper's pattern), every core then runs the row's down.
- Rows: weighted stream-K over clusters (gate/up K-tiles + each row's down paid by its owner); a split row's owner
  can be either side (partial flows toward it): exact balance for R >= ~12, lumpy below.
- No slot credits: the host simulates the program order and sizes the h ring (NS) so a slot is always consumed
  before a peer can overwrite it.
- h exchange is free: a no-sync build (no h traffic) runs at the same speed.

### Steps at M 586 (cycles / expert, tile / useful util)
| step | cyc | tile | useful |
|---|---|---|---|
| first run, owner = head holder | 31,590 | 77.0 | 74.2 |
| balanced (flexible owner) | 32,972 -> 31,512 (no matmul re-init between GU/DN) | 77.2 | 74.4 |
| fused GLU, exp_21f sigmoid | 29,726 | 81.8 | 78.9 |
| SiLU-hiding order + down lag 1 expert (NS 8) | 28,394 | 85.7 | 82.6 |
| down lag 3 rows (NS 8) | 27,348 | 88.9 | 85.7 |
| + GLU constants in programmable regs (glu 3) | 27,281 | 89.1 | 85.9 |
| ceiling: same order, no SiLU at all | 27,023 | 90.0 | 86.7 |
| fixed 4-row ring order (down(q-1) after row q), glu 3 | 28,824 | 84.4 | 81.3 |

Why the order matters: SiLU runs on the PACK thread's SFPU (~2.2-2.7K cycles a row of h) and holds one fp32 DST
half; a down block is only 1.4K cycles (1x4 x K 20; K = I / 32 is the model's), so down blocks right after a row
stall on that half. Rows back to back hide it under the next gate/up row (8.3K); down then runs in batches, which
needs a deeper h ring (NS 8 at M 586; grows with rows per cluster: R 64 -> 13-19, does not fit with resident x / W).
Down consuming freshly produced rows waits on the slowest peer (eager: 46K cycles).

### SiLU on the SFPU (tests/test_glu_bench.py, PACK thread, fp32 DEST, cycles per (gate, up) tile pair)
| variant | cyc | max rel err |
|---|---|---|
| stock silu pass + multiply pass | 1777 | 2e-7 |
| fused one pass, fp32 sigmoid | 1810-1858 | 2e-7 |
| fused, exp_21f + 1 Newton | 1327 | 1.7e-3 |
| same, constants in Prgm regs, clamp 254 (glu 3) | 1103 | 1.7e-3 |
| exp_21f + raw approx recip | 1135 | 6e-3 |
| loads + g*u + store only (floor) | 209-290 | - |
| LUT sigmoid (sigmoid_appx) | 338 | ~1 (2e-2 norm): unusable |
Fusing alone saves nothing; the exp is ~600-740 of it. Hiding a row of h (2 pairs) under one 1.4K down block needs
<= ~700 per pair: not reachable at ~1e-3 accuracy. The hardware approx_exp (SFPARECIP exp mode) is not a full-range
exp: it returns sign(x) e^|x| and saturates at ~4 (tests/test_sfpu_probe.py).
