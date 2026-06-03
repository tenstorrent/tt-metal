# BGE-M3 B8/S512 Latency Optimization (Blackhole P150)

Workflow: tracy → analyze → sweep (if needed) + PCC → optimizations.py → perf.py → repeat.

## Baseline (commit d6d45eb)
- **Forward (trace replay): 48.98 ms** (best 48.95), 163.4 emb/s
- Reference: B1 ≈ 4.1 ms, B32 ≈ 69 ms. Ideal B8 ≈ 8×B1 = 33 ms → ~16 ms slack.

## Tracy bottleneck breakdown (B8, single forward, 48.59 ms device)
| Op | total ms | % | count | µs/call | fidelity |
|----|---------:|--:|------:|--------:|----------|
| Matmul | 28.79 | 59.3 | 96 | 299.9 | **HiFi4 / fp32=1** (unoptimized fallback) |
| SDPA | 8.78 | 18.1 | 24 | 365.9 | **HiFi4 / fp32=1** |
| LayerNorm | 3.85 | 7.9 | 49 | 78.5 | HiFi4 |
| Typecast | 2.75 | 5.7 | 73 | 37.6 | |
| NlpCreateHeads | 2.49 | 5.1 | 24 | 103.9 | |
| NLPConcatHeads | 1.26 | 2.6 | 24 | 52.3 | |
| BinaryNg | 0.58 | 1.2 | 24 | 24.3 | |

Matmul shapes/layer (×24 each): ~583µs (MLP-wi N=4096)=14.0ms, ~280µs (QKV)=6.2ms,
~246µs (AttnOut/MLP-wo)=5.4ms, ~90µs=2.2ms. ALL at HiFi4/fp32-on.

### Root cause
B1 and B32 get LoFi/HiFi2 + fp32_dest_acc_en=False compute kernels; B8 falls through
to the generic HiFi4/fp32=1 default (the `max_batch in (1, 32)` gates skip 8).

## Experiments
| # | change | forward ms | vs base | PCC | status |
|---|--------|-----------:|--------:|-----|--------|
| 0 | baseline | 48.98 | 1.00x | — | keep |
| 1 | LoFi/HiFi2 + fp32-off compute kernels for B8 (qkv, attn_out, mlp_wi/wo, sdpa, layernorm) — commit 0ac7ea0 | 38.35 | 1.28x | 0.948 | **keep** |
| 2 | SDPA q256/k512 chunks + max_cores_per_head_batch=8 (B32-style) — commit 9552705 | 36.86 | 1.33x | 0.947 | **keep** |
| 3 | Tuned MLP-wi program config (mm2d 11x10 ibw8 sub1x4) — swept 3.93x on op vs default linear — commit d382e1c | 30.59 | **1.60x** | 0.941 | **keep** |

### Sweep: B8 MLP-wi (M=4096 K=1024 N=4096 GELU)
Default ttnn.linear routing = 400 µs (terrible for this M). Best mm2d_g11x10_ibw8_sub1x4 = 101.9 µs (3.93x).
Note PCC now 0.941 — close to 0.94 floor. Remaining headroom must watch accuracy.

### Sweep: B8 QKV / AttnOut / MLPwo (DEAD END)
Default ttnn.linear is ALREADY optimal for these 3 (N=3072/1024/1024). All mm2d
variants 0.5-0.8x slower. Unlike MLP-wi (N=4096), the default routing handles
these well. Isolated matmul cost (76/33/104µs) << in-model (185/111/142µs),
so in-model cost is L1/memory pressure, not the matmul config. No win here.

## Tracy after exp 3 (30.35 ms device)
| Op | ms | % | µs/call |
|----|---:|--:|--------:|
| Matmul | 12.27 | 40.4 | 127.8 |
| SDPA | 7.35 | 24.2 | 306.2 |
| LayerNorm | 3.57 | 11.7 | 72.8 |
| Typecast | 2.75 | 9.1 | 37.7 |
| NlpCreateHeads | 2.49 | 8.2 | 103.6 |
| NLPConcatHeads | 1.26 | 4.2 | 52.5 |

Next: Typecast (2.75ms, 73 calls) + heads (3.75ms) are memory-movement.
SDPA 7.35ms already B32-tuned. PCC 0.941 limits further LoFi.

## FINAL RESULT
- **Forward: 48.98 ms → 30.59 ms = 1.60x faster (−18.4 ms)**, PCC 0.941 (≥0.94 gate)
- H2D→Forward→D2H (customer-facing): H2D 0.23ms + Forward 30.63ms + D2H 1.06ms ≈ 31.9ms
- Throughput: 163 → 261 emb/s
- No regression: B1 = 4.11ms (was 4.14), B32 = 61.1ms (unchanged) — changes are batch-8-gated.

### Commits
- 0ac7ea0: LoFi/HiFi2 + fp32-off compute kernels (−10.6ms)
- 9552705: SDPA q256/k512 + max_cores_per_head_batch=8 (−1.5ms)
- d382e1c: tuned MLP-wi program config ibw8 sub1x4 (−6.3ms, 3.9x on op)
- 9539261: batch8 in tracy_perf/perf + sweep harnesses

### Remaining headroom (not pursued — PCC margin thin at 0.941)
- SDPA 7.35ms (already B32-tuned), Typecast 2.75ms + heads 3.75ms (memory-movement, structural).
- LayerNorm sharded config is B1-only; could be ported to B8 but is shape-specific.
- QKV/AttnOut/MLPwo: default ttnn.linear already optimal (swept, no win).

## Tracy after exp 1 (38.0 ms device)
| Op | ms | % | µs/call |
|----|---:|--:|--------:|
| Matmul | 18.51 | 48.7 | 192.8 |
| SDPA | 8.75 | 23.0 | 364.6 |
| LayerNorm | 3.57 | 9.4 | 72.9 |
| Typecast | 2.75 | 7.2 | 37.7 |
| NlpCreateHeads | 2.48 | 6.5 | 103.5 |
| NLPConcatHeads | 1.26 | 3.3 | 52.4 |

Matmul-wi (~404µs×24=9.7ms) still #1 matmul; SDPA still #2 overall.

## Continued (session 2) — pushing toward 20ms
| # | change | forward ms | PCC | commit |
|---|--------|-----------:|-----|--------|
| 5 | qkv_dtype=bf16: drop 72 bf8->bf16 SDPA typecasts | 29.60 | 0.942 | 5bffcd0 |
| 6 | fused qkv/concat head-split kernels (create hg=4, concat hg=16) | 28.94 | 0.942 | 980dc01 |
| 7 | MLP-wi output -> L1 | 26.40 | 0.942 | a3fc6c1 |
| 8 | MLP-wo output -> L1 | 25.98 | 0.942 | bf6517f |
| 9 | attention output -> L1 | 24.56 | 0.942 | 14800ea |
| 10 | SDPA k_chunk 512->256 | 24.30 | 0.944 | bc6e103 |

**48.98 -> 24.30 ms = 2.02x.** PCC 0.944 (>=0.94 gate).

### Dead ends (session 2)
- QKV/AttnOut matmul program configs: default ttnn.linear optimal across grids {8x8,10x8,8x10,11x10}. MLPwo marginal 1.017x not worth it.
- create_heads output -> L1: clashes with SDPA static CBs (program 15).
- B8 sharded LayerNorm: shard_height=512 (16 tiles) overflows L1 vs matmul L1 outputs (CB clash).
- SDPA 64-core (8x8) grid: regresses to 24.89ms (B8's 128 head-batch pairs want 11x10=110 cores).

## Batch 16 optimization (same workflow as B8)
B16 baseline (no batch-specific tuning) = 94.76ms.

| # | change | forward ms | PCC | commit |
|---|--------|-----------:|-----|--------|
| 1 | Enable B8 fidelity gates for B16 (LoFi/HiFi2 kernels, SDPA tune, fused heads, qkv_dtype=bf16) | 70.65 | 0.954 | d28fa61 |
| 2 | Tuned MLP-wi 2D mcast (g11x10 ibw4 sub2x4, 277.7us vs 810us = 2.9x) | 57.80 | 0.950 | 9e82b31 |
| 3 | SDPA k_chunk 512->256 | 54.33 | 0.950 | ca422c7 |
| 4 | Tuned MLPwo (g11x10 ibw8 sub2x1, 1.54x) + AttnOut (1.13x) 2D mcast | 50.84 | 0.948 | 9e200ff |
| - | + end-to-end B16 PCC test | - | pass | 4a506d1 |

**94.76 -> 50.86 ms = 1.86x.** PCC 0.948 (>=0.94 gate). B16 end-to-end PCC test added.

### Dead ends (B16)
- L1 activation/residual outputs: B16 tensors (16-32MB) too big for 1.5MB L1 (CB clash). B8's L1 wins don't transfer.
- B16 sharded LayerNorm: 8192x1024 = 32MB tensor can't fit sharded over 64 cores + CBs.
- LN fp32_dest_acc_en=False: saves 0.5ms but PCC drops to 0.9359 (< 0.94 gate). B16 has less precision headroom than B8.
- QKV explicit 2D mcast: ibw8 overflows L1, ibw4 slower (51.22 vs 50.88) than default routing. N=3072 default is efficient.
- MLP-wi 1D mcast_in1: per_core_N=128 tiles overflows L1 (2.6MB).
- SDPA exp_approx_mode=True: slightly worse (50.92 vs 50.86).
- Head_groups (create/concat): insensitive, current 4/16 optimal.
- SDPA 64-core grid: B16 has 256 head-batch pairs, wants full 11x10 (like B8).

### B16 is bandwidth-bound
Unlike B8 (4MB tensors fit L1), B16's 16-32MB activations are DRAM-resident. The matmul
program-config tuning (MLP-wi/wo/AttnOut) captured the main gains. Remaining time is
genuine DRAM bandwidth on SDPA (14ms) and the large matmuls.

## BREAKTHROUGH: full bf8 SDPA (B16)
| # | change | forward ms | PCC | commit |
|---|--------|-----------:|-----|--------|
| 5 | B16 full bf8 SDPA (score_dtype + qkv_dtype BOTH bf8, like B32) | 38.57 | 0.947 | 8c5ef13 |

**94.76 -> 38.57 ms = 2.46x.** The SDPA was bandwidth-bound on bf16 Q/K/V (16MB each,
read repeatedly across k-chunks). Setting BOTH score_dtype and qkv_dtype to bf8_b halves
that bandwidth AND avoids typecasts. SDPA 14ms->8ms, GenericOp heads 7.8->4ms.

KEY LESSON: experiment #11 (bf8 qkv ALONE) failed because score_dtype stayed bf16, forcing
bf8->bf16 typecasts. The win requires matching BOTH dtypes (the B32 recipe). My earlier
"SDPA is bandwidth-bound, exhausted" conclusion was correct about the cause but wrong about
the fix — bf8 directly halves the bandwidth.

- B8 full-bf8 SDPA = 18.65ms BUT PCC 0.9249 < 0.94 gate (B8 has tighter margin). B8 keeps bf16.
- B16 full-bf8 PCC = 0.9469 (passes), e2e test passes.

## Post-bf8-SDPA exploration (B16, all confirmed optimal/infeasible)
After the full-bf8-SDPA win (38.57ms), re-explored adjacent levers:
- QKV explicit 2D mcast (bf8 output halves CBs): ALL variants >= 38.58 default, ibw8 still L1-overflow. QKV default routing genuinely optimal (confirmed at both bf16 and bf8 output).
- bf8 LN output: ttnn.layer_norm has no dtype kwarg; output dtype = input dtype. Can't change without extra typecast (net negative).
- Side-effect of bf8 SDPA: QKV/AttnOut WEIGHTS also became bf8 (qkv_dtype/output_dtype gate the weight load dtype), cutting QKV matmul 310us->230us. Part of why the win was large.

Final B16 matmul buckets (38.6ms total, Matmul 19.6ms):
- MLP-wi ~280us x24 = 6.69ms (tuned ibw4 sub2x4, L1 ceiling)
- QKV ~230us x25 = 5.68ms (default routing, optimal)
- MLP-wo ~220us x23 = 5.07ms (tuned ibw8 sub2x1)
- AttnOut ~90us x24 = 2.15ms (tuned)

### Final state (all PCC-passing, no regressions)
- B1  = 4.11ms
- B8  = 24.29ms (bf16 SDPA; full-bf8 drops PCC to 0.9249 < gate)
- B16 = 38.60ms (full-bf8 SDPA, PCC 0.947) = 2.46x over 94.76 baseline
- B32 = 61.07ms
