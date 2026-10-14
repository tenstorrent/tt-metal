# SDPA: optional softmax row sums on the math thread (`math_thread_row_sums`)

Split out of #59695. Adds one opt-in flag to `ttnn.transformer.scaled_dot_product_attention`; default `False`,
so nothing changes unless a caller asks for it.

## What it does

With a single K chunk there is no online-softmax correction, so every exp'd score for a row is still sitting in
`cb_qkt_im` when normalize runs. The flag takes each softmax denominator from those scores with a matmul against
a column of ones **on the math thread**, instead of L1-accumulating it per position **on the pack thread**.

Eligibility (silently ignored otherwise): Blackhole, `fp32_dest_acc_en=False` (the streaming kernel), a single K
chunk (`k_chunk_size >= s`), no `attention_sink`. The host gates it and sizes the recip-scratch CB to match
(compute arg 29), so kernel use and CB sizing cannot disagree.

## Why it is opt-in rather than automatic

Whether it wins depends on which thread paces the softmax, and that flips with math fidelity.

Blackhole P150b, 32/8 heads, s=512, non-causal, K/V in L1, `exp_approx_mode=True`, device cycles,
flag off → on (negative = faster with it on):

| config | batch 8 | batch 16 | batch 32 |
|---|---|---|---|
| **LoFi**, d=128, q_chunk 128 | −0.5% | −0.0% | −0.6% |
| HiFi2, d=128, q_chunk 128 | **+6.0%** | **+5.6%** | **+6.5%** |
| HiFi4, d=128, q_chunk 128 | **+8.8%** | **+8.5%** | **+8.7%** |

The flat LoFi row is not a contradiction, it is the point: **this branch has no K/V retention**, so each core
re-reads K/V for every one of its query chunks and at d=128 those calls are bandwidth-bound — there is no
compute left to save. The gain is there as soon as the call is compute-bound again:

| compute-bound config (LoFi) | delta |
|---|---|
| d=**64**, q_chunk 128, batch 16 | **−8.9%** |
| d=128, q_chunk **512**, batch 1 (one chunk per core) | **−6.7%** |
| d=128, q_chunk 256, batch 1, d=64 | −6.9% |
| d=128, q_chunk 256, batch 2 | −5.4% |

For reference, the same batched shapes measured on the older stack *with* `reuse_kv` were −5.3 / −6.6 / −6.4%
at batch 8 / 16 / 32, so these shapes should recover the gain once SDPA grows K/V retention.

The streaming kernel runs the exp on the pack thread, so that thread carries the exp *and* the row-sum packs.
The flag removes the same ~339 cycles of row-sum packs from the pack thread at either fidelity — what changes is
whether the math thread was waiting for them:

| per `sub_exp` call (8 score tiles) | LoFi off → on | HiFi2 off → on |
|---|---|---|
| MATH `SUB` — mostly the `tile_regs_acquire` wait on PACK | **701 → 390** | **293 → 293** |
| MATH `Q@KT MM+Pack` | 1353 → 1367 | 1903 → 1903 |
| PACK `EXP` | 558 → 556 | 608 → 676 |
| PACK `PACK SUB_EXP` (incl. the L1-acc row-sum packs) | **486 → 147** | **479 → 146** |
| MATH `NORM_MATMUL_RECIP` (per row group) | 459 → 1357 | 414 → 1850 |
| MATH `TRISC-KERNEL` | **25456 → 23943** | **30178 → 31915** |

At LoFi the math thread idles 701 cycles per call waiting on PACK, and the flag turns most of that into real
time. At HiFi2 its wait is 293 and **does not move**: its own Q·Kᵀ has grown 1353 → 1903 cycles and is now the
long pole, so the pack thread's saving lands in slack while the normalize matmul the flag adds costs +1436.

Head dim moves it on the same axis, and so does whether the op is compute- or bandwidth-bound:

| axis (LoFi, batch 16, d=128 unless stated) | delta |
|---|---|
| d = 64 / 128 / 256 | **−8.9%** / −0.0% / +0.5% |
| `exp_approx_mode` True / False | −0.0% / +0.4% |
| K/V in L1 vs. all DRAM | −0.0% / −0.3% |

`exp_approx_mode` barely matters because the flag does not take the exp off the pack thread, only the row-sum
packs, and those cost the same either way.

## Accuracy

The row sum accumulates in one 16-bit DEST tile across the K tiles, so its rounding error grows with `s`, where
the pack thread's per-position L1 sums average theirs out. A single call barely moves, but the error is
systematic and accumulates over a model. All five SDXL 1024×1024 module PCC cases, flag forced on:

| test / case | what the flag touches | off | on | bar | on |
|---|---|---|---|---|---|
| `transformermodel` `down_blocks.1` (4096 tok, k512) | 77-tok cross-attn only | 0.9990458 | 0.9990448 | 0.998 | passes |
| `transformermodel` `down_blocks.2` (1024 tok, k1024) | LoFi self-attn | **0.997167** | **0.996846** | 0.997 | **fails** |
| `crossattnmidblock2d` (1024 tok, k1024) | LoFi self-attn | **0.9972232** | **0.9966934** | 0.9968 | **fails** |
| `crossattnupblock` case 0 = `up_blocks.0` (1024 tok, k1024) | HiFi2 self-attn | 0.9897058 | 0.9847438 | 0.968 | passes |
| `crossattnupblock` case 1 = `up_blocks.1` (4096 tok, k512) | 77-tok cross-attn only | 0.9930153 | 0.9928541 | 0.993 | **fails** |

Row 1 is the control: 8 K chunks there, so the flag only reaches that block's 77-token cross-attention and the
block moves 1e-6. Row 5 shows a 77-token cross-attention is not free either — same call as row 1, yet the block
moves 1.6e-4 and crosses its 0.993 bar (it passes by only 1.5e-5 with the flag off, so it is a knife-edge test).

## Measurement method

Latency is reported in **device cycles, not wall-clock µs**. Blackhole's AICLK moves with load and the batch-16
shape sits between two clock states, so the same config re-measured minutes later moved ~9% in µs. Device cycles
do not depend on the clock: the same arm measured twice under two names agreed to 0.03 pp, where the wall-clock
readings of the same two runs disagreed by 6 pp. Each arm is one traced call, 11 blocking replays, first dropped,
median over replays; run-to-run spread 0.2–0.6%. Flag off and on are the same binary.

Tooling is on `amorrison/high-batch-optim` under `models/demos/blackhole/pplx_embed_4b/perf_tools/`
(`bench_cycles.py`, `run_cycles.sh`, `bench_zones.py`, `bench_denom.py`, `sdpa_kernel_variants.py`).

## Test

`tests/ttnn/unit_tests/operations/sdpa/test_sdpa_single_k_chunk.py` puts every masking mode (none, causal, user
mask, windowed, windowed causal), a padded S and an odd-tile Q chunk through one K chunk, against torch with a
per-row scale check (PCC alone misses a wrong denominator) and against the same call in several K chunks. Plus
`test_sdpa_single_k_chunk_full_sync` (full DST sync doubles the normalize row group, so the host must size the
recip scratch for the kernel's group) and `test_sdpa_single_k_chunk_row_sums_off_by_default` (a call without the
flag is bit-identical to one with it off).

47 passed, 3 skipped on a Blackhole P150b.
