# DiffVAE NA tile sparsity (task #213, C2)

CPU calculator for PLAN.md 1.4, P4 (narrowing) and P8 (Q-chunk shape). No device jobs.
Script: `tt-project/t-diffvae/sparsity_calc.py` (window rule imported from
`models/tt_dit/layers/neighborhood_reference.py`).

```
python tt-project/t-diffvae/sparsity_calc.py --markdown \
  --regimes 5:8,2,2:1,1,1:1,1,1 5:2,4,4:1,1,1:1,1,1 5:2,4,4:2,1,1:1,1,1 5:2,4,4:1,1,1:2,4,4
# gather cross-check against the C++ planner (needs a ttnn build, e.g. worktree t158):
LD_LIBRARY_PATH=$T/build_Release/lib TT_METAL_HOME=$T PYTHONPATH=$T:$T/ttnn \
  python tt-project/t-diffvae/sparsity_calc.py --check-planner
```

## Definitions and units

All counts are **tile pairs** (one 32x32 QK^T tile, and the matching PV tile) **per head, per NA layer**.

- **gathered**: query bricks x the gather's slot count. This is what the op does today: every query brick
  multiplies against every slot of its chunk's gather, and masked slots cost the same as live ones.
  The gather extent per axis is the planner's constant: the worst-aligned chunk origin.
- **live**: per query brick, the key bricks holding at least one key inside the window of one of the
  brick's 32 queries. This is what narrowing (P4) would iterate.
- **exact**: real queries x window keys / 1024. It is the dense-equivalent floor with no tile waste.
- **Regime**: per axis, L or H if any query in the brick had its window moved by the low or high volume
  edge, otherwise I. That gives 27 classes. Windows are per-axis products, so every count factors per axis.
- **op today**: `ok` means the op accepts it as-is. `unsafe-chunk` means the Q chunk is bigger than the
  stride, which the op rejects unless `DIFFVAE_NA_UNSAFE_CHUNK=1`.
- **(4,4,4)**: brick (2,4,4) with a 2-brick Q chunk along T (`query_chunk_bricks=(2,1,1)`).
  The other shapes are one-brick chunks.

## Checks

- **Gather**: all 34 accepted configs match `ttnn.transformer.neighborhood_plan` (t158 build, same planner
  source as t210). The match includes today's pinned production gathers: S2 63, S3 45 and S4 27.
- **Live and exact**: brute force over `neighborhood_mask` on 4 small volumes gives identical totals. The
  volumes cover strides 1 and (2,4,4) and bricks (2,4,4), (8,4,1) and (1,4,8).

## Assumptions

- Stage volumes are derived from the latent (19,34,60) and the upsample chain, not read from the checkpoint:
  S1 (21,34,60), S2 (21,68,120), S3 (41,68,120), S4 (81,136,240), S5 (145,272,480).
  blx03 was unreachable over ssh. The volumes are consistent with the pinned test geometries
  (`test_choose_sharded_brick_regression`).
- The tables use the global volume. W sharding over 8 and its halo are left out. On the stride-1 configs,
  a W halo is real keys, so the gather on W is unchanged.
- **Stage 5 runs in T bands today**, and each band's gather is 147 slots: the planner gives (3,7,7) for
  (84|83|72, 272, 480) with brick (8,2,2). The whole-volume row shows 196 = (4,7,7). That extra T brick
  comes from the 1-site ragged tail brick at frame 144 and does not happen in production.
  Banded today: gathered ~91.2M (620,160 bricks x 147), live ~88.7M, exact 24.6M.
  So today wastes 3.70x versus exact, and narrowing would recover only ~1.03x.
- **Stage 1 today** is the linear-order executor. Its row is the sum, over `plan_na3d` tile groups, of
  ceil(Nq/32) x ceil(Nk/32): dense masked SDPA per group, padded to tiles. It is an estimate of tile work,
  not of the kernel's own chunking.

## Answers

1. **Narrowing (P4) has little headroom at chunk (1,1,1), stride 1.** In the interior, the gather is already
   exactly the brick's window union: interior (2,4,4) bricks at S5 are 175/175 live, today's (8,2,2)
   147/147. Narrowing only removes edge slots.
   - S5: 1.03-1.04x for every one-brick shape.
   - S4: 1.04-1.06x.
   - S2/S3 with today's thin bricks (8,4,1)/(16,2,1): 1.34x/1.29x. Their long T brick against a T window
     of 3 makes the worst-case misalignment cost a whole extra brick.

   Narrowing pays only with multi-brick chunks: 1.19x (S5) to 1.5x (S2-S4).
2. **Brick-level waste versus exact is the real gap at stride 1.** S5 needs 147-175 slots per brick against
   41.6 exact, so 3.5-4.2x.
   - Among the requested shapes, none beats today's 147 at S5: (2,2,8) ties at 147, (1,4,8) needs 165 and
     (2,4,4) needs 175.
   - PLAN 1.4's "(2,4,4) is 1.77x exact" counts sites (12x14x14). In slots, which is what the kernel pays,
     it is 7x5x5 = 175, or 4.2x.
   - **(2,4,4) at stride 1 would make S5 ~14% slower than today's (8,2,2) bands**, not faster.
3. **Q chunk (4,4,4) vs (2,4,4) (P8).**
   - At stride 1 the 2-brick T chunk raises the S5 gather from 175 to 200 (+14%) with the same live count.
     It only pays if fetching K/V once per two bricks is worth more than 14% extra matmul, or together with
     narrowing, which makes it compute-neutral.
   - At GNA stride (2,4,4) it is not a legal chunk (chunk must equal the stride), and it grows the gather
     from 54 to 63.
   - On today's op, (4,4,4) is a loss at every stage.
4. **GNA stride (2,4,4) with brick (2,4,4) is the only big lever.**
   - S5: 54 slots per brick, every slot live, 1.31x exact. The windows snap to whole bricks, so the only
     waste is 12^3/11^3.
   - Tile pairs drop to 32.2M from today's ~91.2M (2.83x fewer), and from 104.2M at (2,4,4) stride 1 (3.24x).
   - Narrowing becomes moot.
   - Deterministic stages at (2,4,4) stride (2,4,4): 8 slots and 1.8-3.5x exact.
   - Quality risk is unmeasured. It is PLAN's reserve lever and needs PCC against the unoptimized DiffVAE.
   - Other bricks under the (2,4,4) stride do not snap and gain much less: (1,4,8) 99 slots, (2,2,8) 108.
5. **Deterministic stages 1-4.**
   - (2,4,4) at stride 1 cuts S2 from 63 to 27 slots (2.5x fewer tile pairs) and S3 from 45 to 27 (1.9x).
     S4 stays at 27.
   - But S2/S3 use brick width 1 because W_local = 15 under the W-over-8 shard, so (2,4,4) needs a
     different shard layout there.
   - S1: a bricked (2,4,4) executor would do ~40k tile pairs against ~79k for today's linear-order
     estimate.

## Per-stage table (all regimes summed)
### Stage 1: volume (21, 34, 60), window (3, 7, 7), 4 NA layers

| brick/Q-chunk | stride | gather bricks (t,h,w) | slots | op today | gathered | live | exact | gath/exact | live/exact | gath/live |
|---|---|---|---|---|---|---|---|---|---|---|
| today: linear-order, tile (6, 9, 15) | (1,1,1) | - | - | ok | 79.2k | - | 6,150 | 12.88 | - | - |
| (2,4,4) | (1, 1, 1) | (3, 3, 3) | 27 | ok | 40.1k | 34.7k | 6,150 | 6.52 | 5.64 | 1.16 |
| (2,4,4) | (2, 4, 4) | (2, 3, 2) | 12 | ok | 17.8k | 13.2k | 6,150 | 2.90 | 2.15 | 1.35 |
| (4,4,4) | (1, 1, 1) | (4, 3, 3) | 36 | unsafe-chunk | 53.5k | 34.7k | 6,150 | 8.69 | 5.64 | 1.54 |
| (4,4,4) | (2, 4, 4) | (3, 3, 2) | 18 | unsafe-chunk | 26.7k | 13.2k | 6,150 | 4.35 | 2.15 | 2.02 |
| (1,4,8) | (1, 1, 1) | (3, 3, 3) | 27 | ok | 40.8k | 36.0k | 6,150 | 6.64 | 5.86 | 1.13 |
| (1,4,8) | (2, 4, 4) | (3, 3, 3) | 27 | ok | 40.8k | 27.7k | 6,150 | 6.64 | 4.51 | 1.47 |
| (2,2,8) | (1, 1, 1) | (3, 5, 3) | 45 | ok | 67.3k | 55.2k | 6,150 | 10.95 | 8.98 | 1.22 |
| (2,2,8) | (2, 4, 4) | (2, 4, 3) | 24 | ok | 35.9k | 32.9k | 6,150 | 5.84 | 5.35 | 1.09 |

### Stage 2: volume (21, 68, 120), window (3, 7, 7), 6 NA layers

| brick/Q-chunk | stride | gather bricks (t,h,w) | slots | op today | gathered | live | exact | gath/exact | live/exact | gath/live |
|---|---|---|---|---|---|---|---|---|---|---|
| today (8, 4, 1) | (1, 1, 1) | (3, 3, 7) | 63 | ok | 385.6k | 288.1k | 24.6k | 15.67 | 11.71 | 1.34 |
| (2,4,4) | (1, 1, 1) | (3, 3, 3) | 27 | ok | 151.5k | 133.7k | 24.6k | 6.16 | 5.43 | 1.13 |
| (2,4,4) | (2, 4, 4) | (2, 2, 2) | 8 | ok | 44.9k | 44.9k | 24.6k | 1.82 | 1.82 | 1.00 |
| (4,4,4) | (1, 1, 1) | (4, 3, 3) | 36 | unsafe-chunk | 202.0k | 133.7k | 24.6k | 8.21 | 5.43 | 1.51 |
| (4,4,4) | (2, 4, 4) | (3, 2, 2) | 12 | unsafe-chunk | 67.3k | 44.9k | 24.6k | 2.74 | 1.82 | 1.50 |
| (1,4,8) | (1, 1, 1) | (3, 3, 3) | 27 | ok | 144.6k | 132.7k | 24.6k | 5.88 | 5.40 | 1.09 |
| (1,4,8) | (2, 4, 4) | (3, 2, 3) | 18 | ok | 96.4k | 92.1k | 24.6k | 3.92 | 3.74 | 1.05 |
| (2,2,8) | (1, 1, 1) | (3, 5, 3) | 45 | ok | 252.4k | 221.3k | 24.6k | 10.26 | 9.00 | 1.14 |
| (2,2,8) | (2, 4, 4) | (2, 4, 3) | 24 | ok | 134.6k | 128.7k | 24.6k | 5.47 | 5.23 | 1.05 |

### Stage 3: volume (41, 68, 120), window (3, 5, 5), 4 NA layers

| brick/Q-chunk | stride | gather bricks (t,h,w) | slots | op today | gathered | live | exact | gath/exact | live/exact | gath/live |
|---|---|---|---|---|---|---|---|---|---|---|
| today (16, 2, 1) | (1, 1, 1) | (3, 3, 5) | 45 | ok | 550.8k | 428.4k | 24.5k | 22.48 | 17.48 | 1.29 |
| (2,4,4) | (1, 1, 1) | (3, 3, 3) | 27 | ok | 289.2k | 263.0k | 24.5k | 11.80 | 10.73 | 1.10 |
| (2,4,4) | (2, 4, 4) | (2, 2, 2) | 8 | ok | 85.7k | 85.7k | 24.5k | 3.50 | 3.50 | 1.00 |
| (4,4,4) | (1, 1, 1) | (4, 3, 3) | 36 | unsafe-chunk | 385.6k | 263.0k | 24.5k | 15.73 | 10.73 | 1.47 |
| (4,4,4) | (2, 4, 4) | (3, 2, 2) | 12 | unsafe-chunk | 128.5k | 85.7k | 24.5k | 5.24 | 3.50 | 1.50 |
| (1,4,8) | (1, 1, 1) | (3, 3, 3) | 27 | ok | 282.3k | 259.2k | 24.5k | 11.52 | 10.58 | 1.09 |
| (1,4,8) | (2, 4, 4) | (3, 2, 2) | 12 | ok | 125.5k | 121.3k | 24.5k | 5.12 | 4.95 | 1.03 |
| (2,2,8) | (1, 1, 1) | (3, 3, 3) | 27 | ok | 289.2k | 267.5k | 24.5k | 11.80 | 10.92 | 1.08 |
| (2,2,8) | (2, 4, 4) | (2, 3, 2) | 12 | ok | 128.5k | 124.2k | 24.5k | 5.24 | 5.07 | 1.03 |

### Stage 4: volume (81, 136, 240), window (3, 5, 5), 2 NA layers

| brick/Q-chunk | stride | gather bricks (t,h,w) | slots | op today | gathered | live | exact | gath/exact | live/exact | gath/live |
|---|---|---|---|---|---|---|---|---|---|---|
| today (8, 2, 2) | (1, 1, 1) | (3, 3, 3) | 27 | ok | 2,423.5k | 2,276.6k | 193.6k | 12.52 | 11.76 | 1.06 |
| (2,4,4) | (1, 1, 1) | (3, 3, 3) | 27 | ok | 2,258.3k | 2,153.8k | 193.6k | 11.66 | 11.12 | 1.05 |
| (2,4,4) | (2, 4, 4) | (2, 2, 2) | 8 | ok | 669.1k | 669.1k | 193.6k | 3.46 | 3.46 | 1.00 |
| (4,4,4) | (1, 1, 1) | (4, 3, 3) | 36 | unsafe-chunk | 3,011.0k | 2,153.8k | 193.6k | 15.55 | 11.12 | 1.40 |
| (4,4,4) | (2, 4, 4) | (3, 2, 2) | 12 | unsafe-chunk | 1,003.7k | 669.1k | 193.6k | 5.18 | 3.46 | 1.50 |
| (1,4,8) | (1, 1, 1) | (3, 3, 3) | 27 | ok | 2,230.7k | 2,138.4k | 193.6k | 11.52 | 11.04 | 1.04 |
| (1,4,8) | (2, 4, 4) | (3, 2, 2) | 12 | ok | 991.4k | 974.9k | 193.6k | 5.12 | 5.03 | 1.02 |
| (2,2,8) | (1, 1, 1) | (3, 3, 3) | 27 | ok | 2,258.3k | 2,172.2k | 193.6k | 11.66 | 11.22 | 1.04 |
| (2,2,8) | (2, 4, 4) | (2, 3, 2) | 12 | ok | 1,003.7k | 987.0k | 193.6k | 5.18 | 5.10 | 1.02 |

### Stage 5: volume (145, 272, 480), window (11, 11, 11), 8 NA layers

| brick/Q-chunk | stride | gather bricks (t,h,w) | slots | op today | gathered | live | exact | gath/exact | live/exact | gath/live |
|---|---|---|---|---|---|---|---|---|---|---|
| today (8, 2, 2) | (1, 1, 1) | (4, 7, 7) | 196 | ok | 121,551.4k | 88,681.8k | 24,606.9k | 4.94 | 3.60 | 1.37 |
| today, banded (production) | (1, 1, 1) | (3, 7, 7) | 147 | ok | ~91,163.5k | ~88,681.8k | 24,606.9k | 3.70 | 3.60 | 1.03 |
| (2,4,4) | (1, 1, 1) | (7, 5, 5) | 175 | ok | 104,244.0k | 100,190.0k | 24,606.9k | 4.24 | 4.07 | 1.04 |
| (2,4,4) | (2, 4, 4) | (6, 3, 3) | 54 | ok | 32,166.7k | 32,166.7k | 24,606.9k | 1.31 | 1.31 | 1.00 |
| (4,4,4) | (1, 1, 1) | (8, 5, 5) | 200 | unsafe-chunk | 119,136.0k | 100,190.0k | 24,606.9k | 4.84 | 4.07 | 1.19 |
| (4,4,4) | (2, 4, 4) | (7, 3, 3) | 63 | unsafe-chunk | 37,527.8k | 32,166.7k | 24,606.9k | 1.53 | 1.31 | 1.17 |
| (1,4,8) | (1, 1, 1) | (11, 5, 3) | 165 | ok | 97,614.0k | 94,825.9k | 24,606.9k | 3.97 | 3.85 | 1.03 |
| (1,4,8) | (2, 4, 4) | (11, 3, 3) | 99 | ok | 58,568.4k | 57,917.6k | 24,606.9k | 2.38 | 2.35 | 1.01 |
| (2,2,8) | (1, 1, 1) | (7, 7, 3) | 147 | ok | 87,565.0k | 85,035.9k | 24,606.9k | 3.56 | 3.46 | 1.03 |
| (2,2,8) | (2, 4, 4) | (6, 6, 3) | 108 | ok | 64,333.4k | 63,618.6k | 24,606.9k | 2.61 | 2.59 | 1.01 |


## Stage 5 per-regime tables

Regimes with 0 bricks are omitted. 'exact/brick' is lower in H on T because the last T brick is partly ghost.
Stage 5, brick (8, 2, 2), chunk (1, 1, 1), stride (1, 1, 1), gather (4, 7, 7) = 196 slots

| regime (t,h,w) | query bricks | gathered/brick | live/brick | exact/brick | live/gathered |
|---|---|---|---|---|---|
| III | 486,720 | 196 | 147.0 | 41.6 | 0.75 |
| HII | 60,840 | 196 | 147.0 | 23.4 | 0.75 |
| LII | 30,420 | 196 | 98.0 | 41.6 | 0.50 |
| ILI | 11,232 | 196 | 126.0 | 41.6 | 0.64 |
| IHI | 11,232 | 196 | 126.0 | 41.6 | 0.64 |
| IIL | 6,240 | 196 | 126.0 | 41.6 | 0.64 |
| IIH | 6,240 | 196 | 126.0 | 41.6 | 0.64 |
| HLI | 1,404 | 196 | 126.0 | 23.4 | 0.64 |
| HHI | 1,404 | 196 | 126.0 | 23.4 | 0.64 |
| HIL | 780 | 196 | 126.0 | 23.4 | 0.64 |
| HIH | 780 | 196 | 126.0 | 23.4 | 0.64 |
| LLI | 702 | 196 | 84.0 | 41.6 | 0.43 |
| LHI | 702 | 196 | 84.0 | 41.6 | 0.43 |
| LIL | 390 | 196 | 84.0 | 41.6 | 0.43 |
| LIH | 390 | 196 | 84.0 | 41.6 | 0.43 |
| ILL | 144 | 196 | 108.0 | 41.6 | 0.55 |
| ILH | 144 | 196 | 108.0 | 41.6 | 0.55 |
| IHL | 144 | 196 | 108.0 | 41.6 | 0.55 |
| IHH | 144 | 196 | 108.0 | 41.6 | 0.55 |
| HLL | 18 | 196 | 108.0 | 23.4 | 0.55 |
| HLH | 18 | 196 | 108.0 | 23.4 | 0.55 |
| HHL | 18 | 196 | 108.0 | 23.4 | 0.55 |
| HHH | 18 | 196 | 108.0 | 23.4 | 0.55 |
| LLL | 9 | 196 | 72.0 | 41.6 | 0.37 |
| LLH | 9 | 196 | 72.0 | 41.6 | 0.37 |
| LHL | 9 | 196 | 72.0 | 41.6 | 0.37 |
| LHH | 9 | 196 | 72.0 | 41.6 | 0.37 |

Stage 5, brick (2, 4, 4), chunk (1, 1, 1), stride (1, 1, 1), gather (7, 5, 5) = 175 slots

| regime (t,h,w) | query bricks | gathered/brick | live/brick | exact/brick | live/gathered |
|---|---|---|---|---|---|
| III | 497,408 | 175 | 175.0 | 41.6 | 1.00 |
| LII | 22,272 | 175 | 150.0 | 41.6 | 0.86 |
| HII | 22,272 | 175 | 150.0 | 34.7 | 0.86 |
| ILI | 15,544 | 175 | 122.5 | 41.6 | 0.70 |
| IHI | 15,544 | 175 | 122.5 | 41.6 | 0.70 |
| IIL | 8,576 | 175 | 122.5 | 41.6 | 0.70 |
| IIH | 8,576 | 175 | 122.5 | 41.6 | 0.70 |
| LLI | 696 | 175 | 105.0 | 41.6 | 0.60 |
| LHI | 696 | 175 | 105.0 | 41.6 | 0.60 |
| HLI | 696 | 175 | 105.0 | 34.7 | 0.60 |
| HHI | 696 | 175 | 105.0 | 34.7 | 0.60 |
| LIL | 384 | 175 | 105.0 | 41.6 | 0.60 |
| LIH | 384 | 175 | 105.0 | 41.6 | 0.60 |
| HIL | 384 | 175 | 105.0 | 34.7 | 0.60 |
| HIH | 384 | 175 | 105.0 | 34.7 | 0.60 |
| ILL | 268 | 175 | 85.8 | 41.6 | 0.49 |
| ILH | 268 | 175 | 85.8 | 41.6 | 0.49 |
| IHL | 268 | 175 | 85.8 | 41.6 | 0.49 |
| IHH | 268 | 175 | 85.8 | 41.6 | 0.49 |
| LLL | 12 | 175 | 73.5 | 41.6 | 0.42 |
| LLH | 12 | 175 | 73.5 | 41.6 | 0.42 |
| LHL | 12 | 175 | 73.5 | 41.6 | 0.42 |
| LHH | 12 | 175 | 73.5 | 41.6 | 0.42 |
| HLL | 12 | 175 | 73.5 | 34.7 | 0.42 |
| HLH | 12 | 175 | 73.5 | 34.7 | 0.42 |
| HHL | 12 | 175 | 73.5 | 34.7 | 0.42 |
| HHH | 12 | 175 | 73.5 | 34.7 | 0.42 |

Stage 5, brick (2, 4, 4), chunk (2, 1, 1), stride (1, 1, 1), gather (8, 5, 5) = 200 slots

| regime (t,h,w) | query bricks | gathered/brick | live/brick | exact/brick | live/gathered |
|---|---|---|---|---|---|
| III | 497,408 | 200 | 175.0 | 41.6 | 0.88 |
| LII | 22,272 | 200 | 150.0 | 41.6 | 0.75 |
| HII | 22,272 | 200 | 150.0 | 34.7 | 0.75 |
| ILI | 15,544 | 200 | 122.5 | 41.6 | 0.61 |
| IHI | 15,544 | 200 | 122.5 | 41.6 | 0.61 |
| IIL | 8,576 | 200 | 122.5 | 41.6 | 0.61 |
| IIH | 8,576 | 200 | 122.5 | 41.6 | 0.61 |
| LLI | 696 | 200 | 105.0 | 41.6 | 0.53 |
| LHI | 696 | 200 | 105.0 | 41.6 | 0.53 |
| HLI | 696 | 200 | 105.0 | 34.7 | 0.53 |
| HHI | 696 | 200 | 105.0 | 34.7 | 0.53 |
| LIL | 384 | 200 | 105.0 | 41.6 | 0.53 |
| LIH | 384 | 200 | 105.0 | 41.6 | 0.53 |
| HIL | 384 | 200 | 105.0 | 34.7 | 0.53 |
| HIH | 384 | 200 | 105.0 | 34.7 | 0.53 |
| ILL | 268 | 200 | 85.8 | 41.6 | 0.43 |
| ILH | 268 | 200 | 85.8 | 41.6 | 0.43 |
| IHL | 268 | 200 | 85.8 | 41.6 | 0.43 |
| IHH | 268 | 200 | 85.8 | 41.6 | 0.43 |
| LLL | 12 | 200 | 73.5 | 41.6 | 0.37 |
| LLH | 12 | 200 | 73.5 | 41.6 | 0.37 |
| LHL | 12 | 200 | 73.5 | 41.6 | 0.37 |
| LHH | 12 | 200 | 73.5 | 41.6 | 0.37 |
| HLL | 12 | 200 | 73.5 | 34.7 | 0.37 |
| HLH | 12 | 200 | 73.5 | 34.7 | 0.37 |
| HHL | 12 | 200 | 73.5 | 34.7 | 0.37 |
| HHH | 12 | 200 | 73.5 | 34.7 | 0.37 |

Stage 5, brick (2, 4, 4), chunk (1, 1, 1), stride (2, 4, 4), gather (6, 3, 3) = 54 slots

| regime (t,h,w) | query bricks | gathered/brick | live/brick | exact/brick | live/gathered |
|---|---|---|---|---|---|
| III | 529,584 | 54 | 54.0 | 41.6 | 1.00 |
| HII | 23,364 | 54 | 54.0 | 34.7 | 1.00 |
| LII | 15,576 | 54 | 54.0 | 41.6 | 1.00 |
| ILI | 8,024 | 54 | 54.0 | 41.6 | 1.00 |
| IHI | 8,024 | 54 | 54.0 | 41.6 | 1.00 |
| IIL | 4,488 | 54 | 54.0 | 41.6 | 1.00 |
| IIH | 4,488 | 54 | 54.0 | 41.6 | 1.00 |
| HLI | 354 | 54 | 54.0 | 34.7 | 1.00 |
| HHI | 354 | 54 | 54.0 | 34.7 | 1.00 |
| LLI | 236 | 54 | 54.0 | 41.6 | 1.00 |
| LHI | 236 | 54 | 54.0 | 41.6 | 1.00 |
| HIL | 198 | 54 | 54.0 | 34.7 | 1.00 |
| HIH | 198 | 54 | 54.0 | 34.7 | 1.00 |
| LIL | 132 | 54 | 54.0 | 41.6 | 1.00 |
| LIH | 132 | 54 | 54.0 | 41.6 | 1.00 |
| ILL | 68 | 54 | 54.0 | 41.6 | 1.00 |
| ILH | 68 | 54 | 54.0 | 41.6 | 1.00 |
| IHL | 68 | 54 | 54.0 | 41.6 | 1.00 |
| IHH | 68 | 54 | 54.0 | 41.6 | 1.00 |
| HLL | 3 | 54 | 54.0 | 34.7 | 1.00 |
| HLH | 3 | 54 | 54.0 | 34.7 | 1.00 |
| HHL | 3 | 54 | 54.0 | 34.7 | 1.00 |
| HHH | 3 | 54 | 54.0 | 34.7 | 1.00 |
| LLL | 2 | 54 | 54.0 | 41.6 | 1.00 |
| LLH | 2 | 54 | 54.0 | 41.6 | 1.00 |
| LHL | 2 | 54 | 54.0 | 41.6 | 1.00 |
| LHH | 2 | 54 | 54.0 | 41.6 | 1.00 |
