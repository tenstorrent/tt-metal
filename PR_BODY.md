# SDPA: optional softmax row sums on the math thread (`math_thread_row_sums`)

Split out of #59695. Adds one opt-in flag to `ttnn.transformer.scaled_dot_product_attention`; default `False`,
so nothing changes unless a caller asks for it.

> **Draft — sequenced behind the K/V retention work** (`amorrison/sdpa-gqa-kv-sharing`). Everything below is
> measured with K/V retention **in** (this branch merged with that one), because that is the configuration this
> flag is meant to serve and it lands first. On this branch alone, without retention, each core re-reads K/V for
> every one of its query chunks, the `d=128` batched shapes go bandwidth-bound, and the flag is within 0.5%
> either way; `d=64` still gives −8.9% and one Q chunk per core −6.7%, and the HiFi2/HiFi4 penalty is unchanged.

## What it does

With a single K chunk there is no online-softmax correction, so every exp'd score for a row is still sitting in
`cb_qkt_im` when normalize runs. The flag takes each softmax denominator from those scores with a matmul against
a column of ones **on the math thread**, instead of L1-accumulating it per position **on the pack thread**.

Eligibility (silently ignored otherwise): Blackhole, `fp32_dest_acc_en=False` (the streaming kernel), a single K
chunk (`k_chunk_size >= s`), no `attention_sink`. The host gates it and sizes the recip-scratch CB to match
(compute arg 29), so kernel use and CB sizing cannot disagree.

## Why it is opt-in rather than automatic

Whether it wins depends on which thread paces the softmax, and that flips with math fidelity.

Blackhole P150b, 32/8 heads, s=512, d=128, non-causal, q_chunk 128, K/V in L1 and retained across a core's
chunks, `exp_approx_mode=True`, device cycles, flag off → on (negative = faster with it on):

| fidelity | batch 8 | batch 16 | batch 32 |
|---|---|---|---|
| **LoFi** | **−5.6%** | **−6.3%** | **−6.5%** |
| HiFi2 | +7.2% | +7.2% | +7.4% |
| HiFi4 | +8.6% | +8.6% | +8.8% |

Run-to-run spread is 0.1–0.9%. Other axes, at LoFi and batch 16:

| axis | delta |
|---|---|
| d = 64 / 128 / 256 | **−8.9%** / −6.3% / −2.0% |
| K/V in L1 vs. all DRAM | −6.3% / −2.3% |

Head dim moves it because it sets the math thread's Q·Kᵀ cost per score tile — the thing that decides whether
MATH was waiting on PACK at all. Leaving K/V in DRAM makes the call bandwidth-bound, and what this saves is
compute, so most of the gain goes away.

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
