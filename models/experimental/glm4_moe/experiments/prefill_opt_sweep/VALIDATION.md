# Prefill opt validation (Wormhole Galaxy)

**When:** 2026-07-21 ~01:08–01:36 UTC
**Harness:** `run_validation.sh` (same env as `run_prefill_opt_sweep.sh`, `max_new_tokens=8`)
**Device:** idle before start; Flash then REAP (never concurrent)
**CSVs:** this dir `validation_summary.csv` (REAP) · Flash tree `.../glm4_moe_lite/experiments/prefill_opt_sweep/validation_summary.csv`

## Verdict

| Model | Sweep found useful knobs? | Validation speedup on WH GLX? |
|-------|---------------------------|------------------------------|
| **REAP** | **Yes** — PCM≥1 + CHUNK for batched | **Yes — large.** 128×4 batched **4.8×**; 512×8 batched **28×** |
| **Flash** | Plateau ~15.3s with PCM≥2 or CHUNK≥8192 | **Paired re-run: no PCM1→PCM2 delta.** Sweep’s 28s PCM1/4096 cell **did not reproduce** (re-ran at 15.4s) |

## Flash (4×8, `zai-org/GLM-4.7-Flash`)

| Cell | Role | PCM | CHUNK | prefill_s | Δ vs pair |
|------|------|----:|------:|----------:|----------:|
| 512×8 batched | baseline (was “bad” in sweep) | 1 | 4096 | **15.409** | — |
| 512×8 batched | optimal | 2 | 4096 | **15.404** | **−0.005s (~0%)** |
| 1024×16 batched | baseline | 1 | 4096 | 56.904 | — |
| 1024×16 batched | optimal | 2 | 4096 | 57.267 | +0.36s (noise) |

**Sweep context (same tree, earlier same day):** 512×8 PCM1/4096 = **28.317s** once; all other PCM/CHUNK combos ~15.3s. Validation shows that cliff is **not reliable** — treat as flaky outlier, not a stable PCM1/4096 regression.

**Production / sweep defaults (Flash):** Prefer `GLM4_MOE_LITE_MOE_SPARSE_PREFILL_PCM=2` (or 4) and/or `CHUNK≥8192` for consistency with the sweep plateau. PCM=1/4096 is often fine (~15.4s) but had one bad sample.

## REAP (8×4, `cerebras/GLM-4.7-REAP-218B-A32B`)

| Cell | Role | PCM | CHUNK | prefill_s | Speedup |
|------|------|----:|------:|----------:|--------:|
| 128×4 batched | legacy | 0 | 0 | 22.370 | — |
| 128×4 batched | optimal | 1 | 4096 | **4.666** | **4.79×** |
| 512×8 batched | optimal | 1 | 4096 | **24.317** | — |
| 512×8 batched | legacy | 0 | 0 | **683.112** | opt **28.1×** vs legacy |
| 128×4 serial | legacy | 0 | 0 | **5.018** | best for small serial |
| 128×4 serial | chunked | 1 | 4096 | 5.569 | **1.11× slower** (do not chunk) |

**Production / sweep defaults (REAP):**

- **Batched / multi-user / large T:** `GLM4_MOE_MOE_SPARSE_PREFILL_PCM=1` (or ≥1), `GLM4_MOE_MOE_SPARSE_PREFILL_CHUNK_TOKENS=4096`
- **Small serial / single-user:** `PCM=0`, `CHUNK=0` (legacy full-T)
- Keep attn prefill reduce = **rs_ag** (do not force host)

## Artifacts

- Runner: `run_validation.sh`
- Logs: `val_flash_*.log` (Flash tree), `val_reap_*.log` (this dir)
- Timeline: `validation.log`
