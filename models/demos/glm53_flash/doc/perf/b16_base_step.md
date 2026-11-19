# GLM-5.3-Flash 8x4: the B 16 base decode step at ~1 M (F21, 2026-11-14)

Target: 1 M x 16 users at ~42 tok/s/u = ~23.8 ms per step. Measured: 34.25 ms (29.2 tok/s/u).

## Measured

| row | ISL (per user) | OSL (per user) | concurrency | step |
|---|---|---|---|---|
| ctx_full_p1m_rand (probe, random keys, identical rows, 8 lanes) | 1,048,000 (position) | 20 traced | 16 | 34.25 ms (761 us / layer) |
| ctx_l07_p1m_rand (layers 0-7 = 3 dense + 3 sparse_kda + 2 sparse_dsa) | 1,048,000 | 20 traced | 16 | 6.27 ms |
| mu5_g4k3_1m plain pass (e2e, width 16, 12 idle rows) | 1,048,447 | 128 | 4 | 30.8 ms |
| m5_k3_1m plain (B 4 build) | 1,048,447 | 128 | 1 | 20.9 ms |
| nx_4k_x16 (e2e, distinct users) | 4,096 | 128 | 16 | 31.3 ms |

The sparse_dsa layer 3 at 1 M x 16 (traced, per-chip median, us): index + local select 375, ag1 94, merge 271, att lsel 62, pass0 67, rs1 wait 363, rs2 wait 32, attention 740.

History (2026-11-10 probe matrix, pos 127, identical rows, older tree): the t 4 -> t 16 growth per layer was dense_kda +297 us, sparse_kda +168, sparse_dsa +164 = +7.9 ms per step. Since then:
- KDA user groups 16 at t 16: -1.0 ms.
- GLM53_MKX_XROWS_COAL: -2.8 ms.

Distinct users add ~3 ms at B 16 (the routed-expert union: e2e vs the identical-row probe).

## Estimated split at 1 M x 16 (identical rows; +-20 %)

| kind | per layer | count | per step |
|---|---|---|---|
| sparse_dsa | ~1.49 ms | 11 | ~16.4 ms |
| sparse_kda | ~0.50 ms | 31 | ~15.5 ms |
| dense_kda | ~0.60 ms | 3 | ~1.8 ms |

Where the t = 16 cost sits:
- **Per-row work on all 16 rows, live or idle.** This covers the MoE (expert union, x rows, AR payload t x 8 KB), mHC (16-row Sinkhorn / collapse: ~35-45 us per attention site at t 16 vs ~21 at t 1) and the KDA recurrence.
- **The DSA per-user work at long context.** Each of the 8 lanes runs 2 tokens in series at t 16; 16 lanes cap at 8 on the B 16 build's core budget (14 free cores).

Taking one point of padding cost: 1 M x 4 at width 16 (30.8 ms) vs narrow width 4 (~21 ms) -> ~0.7 ms per idle row.

## Ranked levers

| lever | 1 M x 16 | 1 M x 8 | status |
|---|---|---|---|
| native width 8 at 5-8 live (no 8 idle rows) | 0 | est. -4..-6 ms | ported here (320cc81b1c4); chain_b16_1 B |
| DSA merge pre-filter (F19) | est. -0.8..-2.4 ms | -0.4..-1.2 | chain_pf2_1 |
| 16 lanes (index + local select 2 tokens per lane -> 1) | est. -1.7..-2.1 ms | ~0 | needs 2 more free cores (not contained) |
| routed-expert union bytes (distinct users) | ~+3 ms to win back | ~+2 | measure first (GLM53_MK_PROBE_DISTINCT); predicted-expert prefetch / overlap |
| mHC FPU collapse (tf32, not bitwise) | est. -0.5..-1 ms | -0.3..-0.6 | existing knob; timing in chain_b16_1 B, accuracy in C |
| sparse_kda per-row residue (mHC, AR bytes, SEM skew) | ~2.5 ms over 31 layers | ~1.2 | no single lever |
| DSA attention 740 us / layer at 16 users | ? | ? | owner gather of 16 x 2048 selected rows: profile (STACK_PROBE marks) |

Even if every estimate lands, x 16 reaches ~28-29 ms (~35 tok/s/u). Closing to 23.8 ms needs the DSA attention / gather and the per-row costs to shrink as well; chain_b16_1's per-kind rows say which.
