# row_exchange: the hybrid MoE row exchange as one fabric program per direction

Replaces, at decode shapes (T users per mesh row, rows x T <= 32), the composites `TtRoutedExperts._gather_rows_compact`
(untilize + all_broadcast + concat + tilize, x and the routing matrix: 8 ops, 113 us per layer at 4 users per row on the
8x4) and `_combine_rows_allreduce` (reduce-scatter + all-gather + selector matmul: 3 ops, 74 us) with one generic op
each, in-program fabric unicasts between the chips of a mesh column (the rows), after the DeepSeek-V3 b1 fused ops'
protocol (`models/demos/deepseek_v3_b1/micro_ops/reduce_to_one_b1`).

| mode | participants | input per chip | output per chip |
|---|---|---|---|
| gather | the rows of a column | `[1,1,32,W]` bf16 TILE, rows 0..T-1 = this row's users (x, routing) | `[1,1,32,W]`: row `r*T+t` = mesh row r's user t, rows past rows*T zero |
| reduce | the rows of a column | `[1,1,32,W]`: row `r*T+t` = this chip's partial for row r's user t | `[1,1,32,W]`: rows 0..T-1 = the sum over the rows of this row's users' partials (fp32, rounded once), rest zero |
| allreduce | the columns of a row | `[1,1,32,W]`: rows 0..T-1 = this chip's TP partial | `[1,1,32,W]`: rows 0..T-1 = the sum over the columns (fp32, rounded once), rest zero; replicated |

The all-reduce runs in two phases inside the one program: column c owns the chunks `[c*cpo, (c+1)*cpo)` (2 of the 8
512-column chunks of H = 4096), receives the other columns' T-row pieces of them (phase 1: one packet per peer),
sums them like the reduce (selector matmul, fp32 DST), writes its chunks to the output and sends the summed rows to
every peer (phase 2: one packet per peer), whose non-owner workers write them (rows past T zero). The model takes it
through `GLMMesh.all_reduce_cols` under `GLM53_TP_ALLREDUCE=fabric` for decode shapes (`cols * T <= 32`), one
instance per call site (`tag`): a peer's next call of one site can only land after this chip's intervening call of
the other site, so the persistent receive staging is never overwritten before it is consumed.

Cores: one exchange core (fabric endpoint, CB_SEND staging) + one worker core per column chunk of <= 16 tiles
(x: 8 chunks of 512, routing: 1 chunk of 288). Packets: `cpp` pieces of `ps` bytes (T rows of a chunk, 4 KB at T = 4)
per 12 KB payload, `np` packets per peer, each a fused unicast write + atomic increment into the peer's persistent
receive staging (DRAM, one page per (source row, packet)) and arrival semaphore (one global semaphore per instance).
Unit: `tests/custom_ops/test_row_exchange.py` (per-chip exactness vs torch on the 8x4 for the three modes, the (2,4)
block with padding, trace replay, the op's wall time: gather 40 us, reduce 42 us traced on the 8x4 at T = 4).
