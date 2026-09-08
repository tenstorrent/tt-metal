# Qwen3.6-27B (P300X2) serving benchmark — optimized vs baseline

tt-inference-server `--workflow benchmarks` (`vllm bench serve`), concurrency 1 and 32.
- **baseline**: reference build.
- **optimized**: prefill kernels + serving-path fixes + batched fused GDN decode op (`QWEN36_GDN_DECODE_FUSED=2`).

All times in ms. Speedup = baseline / optimized (higher is better).

| ISL | OSL | conc | TTFT base | TTFT opt | TTFT x | TPOT base | TPOT opt | TPOT x | E2EL base | E2EL opt | E2EL x |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 128 | 128 | 1 | 323 | 94 | 3.43x | 41.4 | 30.5 | 1.36x | 5575 | 3964 | 1.41x |
| 128 | 128 | 32 | 11488 | 2728 | 4.21x | 95.7 | 44.3 | 2.16x | 23636 | 8351 | 2.83x |
| 128 | 1024 | 1 | 323 | 97 | 3.33x | 41.4 | 30.6 | 1.35x | 42672 | 31368 | 1.36x |
| 128 | 1024 | 32 | 9568 | 2734 | 3.50x | 86.8 | 41.0 | 2.12x | 98365 | 44682 | 2.20x |
| 1024 | 128 | 1 | 363 | 187 | 1.94x | 41.4 | 30.6 | 1.35x | 5625 | 4078 | 1.38x |
| 1024 | 128 | 32 | 10623 | 5489 | 1.94x | 98.3 | 45.8 | 2.15x | 23106 | 11308 | 2.04x |
| 2048 | 128 | 1 | 480 | 267 | 1.80x | 41.5 | 30.6 | 1.36x | 5751 | 4158 | 1.38x |
| 2048 | 128 | 32 | 14735 | 7931 | 1.86x | 95.7 | 47.7 | 2.01x | 26892 | 13993 | 1.92x |
| 4096 | 128 | 1 | 814 | 519 | 1.57x | 41.7 | 30.9 | 1.35x | 6115 | 4437 | 1.38x |
| 4096 | 128 | 32 | 24926 | 15706 | 1.59x | 102.1 | 52.4 | 1.95x | 37892 | 22363 | 1.69x |
| 8192 | 128 | 1 | 1520 | 1054 | 1.44x | 42.0 | 31.0 | 1.35x | 6857 | 4996 | 1.37x |
| 8192 | 128 | 32 | 47161 | 31749 | 1.49x | 111.2 | 67.8 | 1.64x | 61282 | 40364 | 1.52x |
| 16384 | 128 | 1 | 3059 | 2127 | 1.44x | 42.2 | 31.5 | 1.34x | 8419 | 6128 | 1.37x |
| 16384 | 128 | 31 | 69441 | 49044 | 1.42x | 304.0 | 230.9 | 1.32x | 108046 | 78364 | 1.38x |
| 32768 | 128 | 1 | 6569 | 4566 | 1.44x | 42.9 | 32.2 | 1.33x | 12021 | 8654 | 1.39x |
| 32768 | 128 | 15 | 70519 | 49801 | 1.42x | 302.4 | 244.9 | 1.23x | 108922 | 80909 | 1.35x |
| 65536 | 128 | 1 | 15348 | 10322 | 1.49x | 44.4 | 34.9 | 1.27x | 20987 | 14758 | 1.42x |
| 65536 | 128 | 8 | 85432 | 59240 | 1.44x | 362.4 | 301.2 | 1.20x | 131457 | 97489 | 1.35x |
| 131072 | 128 | 1 | 39829 | 25116 | 1.59x | 47.0 | 49.7 | 0.95x | 45799 | 31432 | 1.46x |
| 131072 | 128 | 4 | 109032 | 73378 | 1.49x | 462.8 | 366.4 | 1.26x | 167810 | 119908 | 1.40x |

Summary vs baseline: TTFT 1.4–4.2x, TPOT 1.2–2.2x, E2EL 1.35–2.8x across the sweep. The batched fused
GDN decode op drives the concurrency-32 TPOT wins (2.0–2.2x for prompts up to ~2k). Single exception:
131072/OSL128 at concurrency 1, TPOT is 5% slower (47.0 → 49.7 ms) — that regime is attention/KV-bound,
not GDN-bound, and the fused op's state handling adds slight overhead; E2EL is still 1.46x better there.

Output quality (8 concurrent real prompts, greedy): coherent and factually correct (7/8), comparable to
baseline, with mild greedy-decode degradation on 1 prompt (bf16 rounding accumulation); not garbage.

Optimizations are env-gated (default off): the serving set is
`QWEN36_GDN_OUT_MODE=agmm QWEN36_GDN_CONV=kda TT_SDPA_GQA_MCAST=1 QWEN36_SDPA_K_CHUNK=256`
`QWEN36_AGMM_LAYOUT=nt11x8 TT_GDN_SCAN_MCAST=1 TT_SDPA_GQA_MCAST_QPAIR=1 QWEN36_GDN_PROJ_CHUNKS=1`
`QWEN36_GDN_GB_BF16=1 QWEN36_KDA_TILE_IN=1 QWEN36_AGMM_BARRIER=1 QWEN36_GDN_SLOT_DEVICE_COPY=2`
`QWEN36_PREFILL_LOGITS_FAST=1 QWEN36_PREFILL_BUCKET_TRACE=1 QWEN36_GDN_DECODE_FUSED=2`.

## bf8 paged KV cache (QWEN_SDPA_BF8=1) — long-context decode

bf8 KV halves the paged-cache read that dominates long-context decode. Validated correct (contract decode PCC 0.9999;
4k/16k coherent; 64k extractive retrieval tracks bf16 word-for-word). Added to the optimized build for long context.

bf8+fused vs the same baseline (long-context points; short/mid context unchanged from the fused table above):

| ISL | OSL | conc | TPOT base | TPOT opt+bf8 | TPOT x | TTFT x | E2EL x |
|---|---|---|---|---|---|---|---|
| 8192 | 128 | 32 | 111.2 | 56.2 | 1.98x | 1.52x | 1.60x |
| 16384 | 128 | 1 | 42.2 | 31.1 | 1.36x | 1.47x | 1.40x |
| 16384 | 128 | 31 | 304.0 | 200.2 | 1.52x | 1.48x | 1.49x |
| 32768 | 128 | 15 | 302.4 | 209.2 | 1.45x | 1.49x | 1.47x |
| 65536 | 128 | 8 | 362.4 | 254.8 | 1.42x | 1.56x | 1.51x |
| 131072 | 128 | 1 | 47.0 | 35.7 | 1.32x | 1.70x | 1.64x |
| 131072 | 128 | 4 | 462.8 | 310.6 | 1.49x | 1.67x | 1.60x |

bf8 vs fused-without-bf8 (isolated bf8 gain): 1.15–1.21x TPOT at long-context concurrency; 1.39x at 131072/conc-1. It also
turns the previous 131072/conc-1 fused regression (0.95x vs baseline) into a 1.32x win, so no separate long-context
fused-disable is needed. bf8 KV is TP-only and default off (QWEN_SDPA_BF8=1).
