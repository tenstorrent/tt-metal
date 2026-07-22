# DiffusionGemma prefill + denoise speed vs context (2026-07-22)

Device sweep of the **current** full-depth serving path across prompt-length (context).
Data: `context_speed_sweep_20260722_msl65536.json`.

- Hardware: QB2 / P150x4 (4× Blackhole p300c), mesh (1,4), TP=4.
- Model: full 30-layer 26B-A4B, bf16 weights/KV, `max_seq_len=65536` (one build, 18.8s, 17.3 GiB DRAM).
- Config: **eager** path (no trace flags), production perf profile
  `DG_SPARSE_MOE=1 DG_SPARSE_MOE_TUNED=1 DG_DEDUP_ARGMAX=1`, argmax sampler, canvas 256, 2 blocks/context (steady = block 1). Synchronized device wall time.
- Two denoise configs measured per context: `earlyhalt` (K=48 cap, model's own halt) and `fixed4` (K=4).

## Prefill (per context)

| context | prefill_s (cold) | prefill_s (warm) | warm tok/s |
|--:|--:|--:|--:|
| 256   | 1.34  | 0.90  | 286   |
| 1024  | 3.87  | 1.11  | 919   |
| 4096  | 3.66  | 1.79  | 2,285 |
| 16384 | 7.63  | 5.29  | 3,095 |
| 32768 | 12.48 | 10.98 | 2,984 |

Cold = first prefill of a new shape (includes kernel compile); warm = second. Chunked ragged
prefill (`DG_PREFILL_RAGGED_LONG`) amortizes well — tok/s climbs to ~3k then flattens; 32K warm ≈ 11 s.

## Denoise (per context)

| context | denoise ms/step | commit s | K=4 block / tok·s | K=48 block / tok·s |
|--:|--:|--:|--:|--:|
| 256   | 463 | 1.16 | 3.01 s / 84.9 | 23.4 s / 10.9 |
| 1024  | 477 | 1.10 | 3.01 s / 85.1 | 24.0 s / 10.7 |
| 4096  | 540 | 0.76 | 2.92 s / 87.8 | 26.7 s / 9.6  |
| 16384 | 699 | 1.05 | 3.85 s / 66.5 | 34.6 s / 7.4  |
| 32768 | 939 | 1.37 | 5.13 s / 49.9 | 46.4 s / 5.5  |

`ms/step` and `commit` are a two-point fit from `block(48)` and `block(4)`:
`block(K) ≈ commit + K·step`. `tok·s = 256 / block`.

## Findings

1. **Denoise step has two regimes.** Flat ~465–540 ms/step up to 4096 (prefix small vs the 256
   canvas), then rises with prefix cross-attention: 16384 → ~700, 32768 → ~940 ms/step. From
   4096→32768 (8× context) the step grows ~1.7× — sub-linear but clearly climbing.
2. **~9× faster than the old projection.** work_log §3c projected ~4175 ms/step at full 30L; the
   current path measures **~465 ms/step** at small context (sparse-MoE tuned + dedup argmax + landed
   opts). Commit is **~1 s (batched)**, not the old ~31 s sequential.
3. **Prefill scales well** to ~3k tok/s; 32K prefill ≈ 11 s warm.
4. **Caveat — early-halt did NOT fire (`halted=false` everywhere).** The sweep uses synthetic
   deterministic prefix tokens (so shorter prompts are true prefixes); on non-coherent tokens the
   trajectory never converges, so the K=48 column is the **full-budget worst case**, not real
   early-halt speed. Real coherent prompts halt ~9–17 steps (memory `dg-serving-speed-early-halt`),
   giving, via `block(K) ≈ commit + K·step`: short-context K≈9 → ~5.3 s → ~48 tok/s; K≈17 → ~9 s →
   ~28 tok/s. A coherent-prompt single-point run is the follow-up to confirm.

## Reproduce

```
DG_SPARSE_MOE=1 DG_SPARSE_MOE_TUNED=1 DG_DEDUP_ARGMAX=1 PYTHONPATH=/home/zni/tt-metal \
python -u context_speed_sweep.py --max-seq-len 65536 \
  --prompt-lengths 256,1024,4096,16384,32768 --output <out>.json
```
