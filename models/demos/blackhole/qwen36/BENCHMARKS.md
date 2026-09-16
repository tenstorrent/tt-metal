# Qwen3.6-27B (P300X2) serving benchmark — optimized vs baseline

tt-inference-server `--workflow benchmarks` (`vllm bench serve`), concurrency 1 and up to 32.
- **baseline**: reference build (all optimizations off) — prior recorded reference.
- **optimized**: prefill kernels + serving-path fixes + batched fused GDN decode op (`QWEN36_GDN_DECODE_FUSED=2`).

Optimized numbers **re-measured live 2026-09-10** on the committed build (`c6deaa03308`, bf16 KV). The
baseline column is the previously recorded reference build (not re-run; it is optimization-off and
deterministic). All times in ms. Speedup = baseline / optimized (higher is better).

| ISL | OSL | conc | TTFT base | TTFT opt | TTFT x | TPOT base | TPOT opt | TPOT x | E2EL base | E2EL opt | E2EL x |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 128 | 128 | 1 | 323 | 94 | 3.44x | 41.4 | 30.4 | 1.36x | 5575 | 3950 | 1.41x |
| 128 | 128 | 32 | 11488 | 2721 | 4.22x | 95.7 | 43.5 | 2.20x | 23636 | 8251 | 2.86x |
| 128 | 1024 | 1 | 323 | 96 | 3.36x | 41.4 | 30.7 | 1.35x | 42672 | 31464 | 1.36x |
| 128 | 1024 | 32 | 9568 | 2737 | 3.50x | 86.8 | 41.0 | 2.12x | 98365 | 44668 | 2.20x |
| 1024 | 128 | 1 | 363 | 187 | 1.94x | 41.4 | 30.5 | 1.36x | 5625 | 4062 | 1.38x |
| 1024 | 128 | 32 | 10623 | 5448 | 1.95x | 98.3 | 45.8 | 2.15x | 23106 | 11266 | 2.05x |
| 2048 | 128 | 1 | 480 | 260 | 1.84x | 41.5 | 31.2 | 1.33x | 5751 | 4217 | 1.36x |
| 2048 | 128 | 32 | 14735 | 7865 | 1.87x | 95.7 | 47.7 | 2.01x | 26892 | 13919 | 1.93x |
| 4096 | 128 | 1 | 814 | 515 | 1.58x | 41.7 | 30.9 | 1.35x | 6115 | 4441 | 1.38x |
| 4096 | 128 | 32 | 24926 | 15548 | 1.60x | 102.1 | 52.2 | 1.96x | 37892 | 22174 | 1.71x |
| 8192 | 128 | 1 | 1520 | 1029 | 1.48x | 42.0 | 31.0 | 1.35x | 6857 | 4967 | 1.38x |
| 8192 | 128 | 32 | 47161 | 31473 | 1.50x | 111.2 | 61.3 | 1.81x | 61282 | 39262 | 1.56x |
| 16384 | 128 | 1 | 3059 | 2101 | 1.46x | 42.2 | 31.6 | 1.34x | 8419 | 6112 | 1.38x |
| 16384 | 128 | 31 | 69441 | 47575 | 1.46x | 304.0 | 198.2 | 1.53x | 108046 | 72749 | 1.49x |
| 32768 | 128 | 1 | 6569 | 4566 | 1.44x | 42.9 | 32.1 | 1.34x | 12021 | 8637 | 1.39x |
| 32768 | 128 | 15 | 70519 | 48374 | 1.46x | 302.4 | 203.0 | 1.49x | 108922 | 74157 | 1.47x |
| 65536 | 128 | 1 | 15348 | 10160 | 1.51x | 44.4 | 33.5 | 1.33x | 20987 | 14408 | 1.46x |
| 65536 | 128 | 8 | 85432 | 56691 | 1.51x | 362.4 | 246.4 | 1.47x | 131457 | 87990 | 1.49x |
| 131072 | 128 | 1 | 39829 | 24876 | 1.60x | 47.0 | 36.2 | 1.30x | 45799 | 29470 | 1.55x |
| 131072 | 128 | 4 | 109032 | 67757 | 1.61x | 462.8 | 299.1 | 1.55x | 167810 | 105738 | 1.59x |

Summary vs baseline: TTFT 1.4–4.2x, TPOT 1.3–2.2x, E2EL 1.36–2.86x across the sweep. The batched fused
GDN decode op drives the concurrency-32 TPOT wins (2.0–2.2x for prompts up to ~2k). The earlier
long-context fused-decode regression is **resolved** in this build: 131072/OSL128 at concurrency 1 is now
36.2 ms (1.30x vs the 47.0 ms baseline), where a prior measurement showed a 0.95x slowdown. Long-context
TPOT improved 14–27% vs the previous recorded numbers (e.g. 16384/conc-31 304→198 ms, 65536/conc-8
362→246 ms) at the same bf16-KV config.

## Full optimized sweep (measured 2026-09-10, all points)

TPOT/E2EL in ms; throughput in tokens/s (Tput) and requests/s (Req Tput).

| conc | num | ISL | OSL | TTFT | TPOT | E2EL | Tput out (TPS) | Tput total (TPS) | Req Tput (RPS) |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 8 | 128 | 128 | 94 | 30.4 | 3950 | 32.4 | 65 | 0.253 |
| 32 | 256 | 128 | 128 | 2721 | 43.5 | 8251 | 496.4 | 993 | 3.878 |
| 1 | 4 | 128 | 1024 | 96 | 30.7 | 31464 | 32.5 | 37 | 0.032 |
| 32 | 128 | 128 | 1024 | 2737 | 41.0 | 44668 | 733.6 | 825 | 0.716 |
| 1 | 4 | 1024 | 128 | 187 | 30.5 | 4062 | 31.5 | 284 | 0.246 |
| 32 | 128 | 1024 | 128 | 5448 | 45.8 | 11266 | 363.5 | 3272 | 2.840 |
| 1 | 4 | 2048 | 128 | 260 | 31.2 | 4217 | 30.3 | 516 | 0.237 |
| 32 | 128 | 2048 | 128 | 7865 | 47.7 | 13919 | 294.2 | 5002 | 2.299 |
| 1 | 4 | 4096 | 128 | 515 | 30.9 | 4441 | 28.8 | 951 | 0.225 |
| 32 | 128 | 4096 | 128 | 15548 | 52.2 | 22174 | 184.7 | 6095 | 1.443 |
| 1 | 2 | 8192 | 128 | 1029 | 31.0 | 4967 | 25.8 | 1675 | 0.201 |
| 32 | 64 | 8192 | 128 | 31473 | 61.3 | 39262 | 104.3 | 6780 | 0.815 |
| 1 | 2 | 8192 | 1024 | 1046 | 31.1 | 32860 | 31.2 | 280 | 0.030 |
| 32 | 64 | 8192 | 1024 | 31326 | 52.9 | 85431 | 383.5 | 3452 | 0.375 |
| 1 | 2 | 10000 | 1024 | 1327 | 31.2 | 33207 | 30.8 | 332 | 0.030 |
| 32 | 64 | 10000 | 1024 | 35407 | 60.5 | 97334 | 336.6 | 3624 | 0.329 |
| 1 | 2 | 16384 | 128 | 2101 | 31.6 | 6112 | 20.9 | 2701 | 0.164 |
| 31 | 62 | 16384 | 128 | 47575 | 198.2 | 72749 | 54.5 | 7036 | 0.426 |
| 1 | 1 | 32768 | 128 | 4566 | 32.1 | 8637 | 14.8 | 3809 | 0.116 |
| 15 | 15 | 32768 | 128 | 48374 | 203.0 | 74157 | 25.9 | 6654 | 0.202 |
| 1 | 1 | 65536 | 128 | 10160 | 33.5 | 14408 | 8.9 | 4557 | 0.069 |
| 8 | 8 | 65536 | 128 | 56691 | 246.4 | 87990 | 11.6 | 5970 | 0.091 |
| 1 | 1 | 131072 | 128 | 24876 | 36.2 | 29470 | 4.3 | 4452 | 0.034 |
| 4 | 4 | 131072 | 128 | 67757 | 299.1 | 105738 | 4.8 | 4963 | 0.038 |

Optimizations are **ON BY DEFAULT** for this model (set via `model_config.py` setdefault; override any flag from the environment to disable it). The serving set is
`QWEN36_GDN_OUT_MODE=agmm QWEN36_GDN_CONV=kda TT_SDPA_GQA_MCAST=1 QWEN36_SDPA_K_CHUNK=256`
`QWEN36_AGMM_LAYOUT=nt11x8 TT_GDN_SCAN_MCAST=1 TT_SDPA_GQA_MCAST_QPAIR=1 QWEN36_GDN_PROJ_CHUNKS=1`
`QWEN36_GDN_GB_BF16=1 QWEN36_KDA_TILE_IN=1 QWEN36_AGMM_BARRIER=1 QWEN36_GDN_SLOT_DEVICE_COPY=2`
`QWEN36_PREFILL_LOGITS_FAST=1 QWEN36_PREFILL_BUCKET_TRACE=1 QWEN36_GDN_DECODE_FUSED=2`.

Output quality (8 concurrent real prompts, greedy): coherent and factually correct (7/8), comparable to
baseline, with mild greedy-decode degradation on 1 prompt (bf16 rounding accumulation); not garbage.

## bf8 paged KV cache (QWEN_SDPA_BF8=1) — long-context decode

bf8 KV halves the paged-cache read/footprint at long context. Validated correct (contract decode PCC 0.9999;
4k/16k coherent; 64k extractive retrieval tracks bf16 word-for-word). bf8+fused vs the same baseline
(long-context points; bf8 config not re-run 2026-09-10):

| ISL | OSL | conc | TPOT base | TPOT opt+bf8 | TPOT x | TTFT x | E2EL x |
|---|---|---|---|---|---|---|---|
| 8192 | 128 | 32 | 111.2 | 56.2 | 1.98x | 1.52x | 1.60x |
| 16384 | 128 | 31 | 304.0 | 200.2 | 1.52x | 1.48x | 1.49x |
| 32768 | 128 | 15 | 302.4 | 209.2 | 1.45x | 1.49x | 1.47x |
| 65536 | 128 | 8 | 362.4 | 254.8 | 1.42x | 1.56x | 1.51x |
| 131072 | 128 | 1 | 47.0 | 35.7 | 1.32x | 1.70x | 1.64x |
| 131072 | 128 | 4 | 462.8 | 310.6 | 1.49x | 1.67x | 1.60x |

NOTE (2026-09-10): with the long-context regression now fixed, the **bf16-KV** optimized build already
reaches these long-context TPOTs (e.g. 131072/conc-1 36.2 ms bf16 vs 35.7 ms bf8; 16384/conc-31 198 vs
200 ms). So bf8's isolated **decode-speed** gain at long context is now marginal; its remaining value is
halving the KV DRAM footprint (concurrency/context headroom). bf8 KV is TP-only and default off
(`QWEN_SDPA_BF8=1`).
