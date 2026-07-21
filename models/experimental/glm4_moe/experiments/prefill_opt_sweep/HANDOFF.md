# HANDOFF: GLM-4.7 REAP + Flash Prefill Opt (Wormhole Galaxy)

**Canonical copy:** this file (REAP tree).
**Flash duplicate:** `/home/tt-admin/sdawle/glm47_flash_wh_glx/tt-metal/models/experimental/glm4_moe_lite/experiments/prefill_opt_sweep/HANDOFF.md` (same content).
**Written:** 2026-07-20 ~23:50 UTC. **Updated:** 2026-07-21 after defaults bake + full REAP ISL×B sweep.
Flash sweep **DONE**. Validation **DONE**. Full REAP grid **DONE** (17/20 ok, 3 OOM at T≥65536).
**Speed-up tables:** `SPEEDUPS.md` · **Full compare:** `REAP_FULL_SWEEP_COMPARE.md` · CSV `reap_full_new_summary.csv`.

---

## 1. Goal / context

Speed up **batched and serial prefill** for:

| Variant | Tree | Package | Mesh | Model ID |
|--------|------|---------|------|----------|
| **REAP** | `/home/tt-admin/sdawle/glm47_reap_218b/tt-metal` | `glm4_moe` | **8×4** | `cerebras/GLM-4.7-REAP-218B-A32B` |
| **Flash** | `/home/tt-admin/sdawle/glm47_flash_wh_glx/tt-metal` | `glm4_moe_lite` | **4×8** | `zai-org/GLM-4.7-Flash` |

Both share one Galaxy. Work so far: port Flash-style MoE sparse PCM/chunking into REAP, sweep PCM×CHUNK, attribute speedups vs attn / fill_cache / MoE.

---

## 2. Trees & commands

Always use that tree’s `python_env` and set `TT_METAL_HOME` + `PYTHONPATH` to the same tree.

```bash
# REAP
REAP=/home/tt-admin/sdawle/glm47_reap_218b/tt-metal
cd "$REAP"
export TT_METAL_HOME="$REAP" PYTHONPATH="$REAP"
./python_env/bin/python3 models/experimental/glm4_moe/scripts/debug_run_full_tt_greedy.py \
  --mesh-rows 8 --mesh-cols 4 \
  --model-id cerebras/GLM-4.7-REAP-218B-A32B \
  ...

# Flash
FLASH=/home/tt-admin/sdawle/glm47_flash_wh_glx/tt-metal
cd "$FLASH"
export TT_METAL_HOME="$FLASH" PYTHONPATH="$FLASH"
./python_env/bin/python3 models/experimental/glm4_moe_lite/scripts/debug_run_full_tt_greedy.py \
  --mesh-rows 4 --mesh-cols 8 --kv-cache-dtype bf8 \
  --model-id zai-org/GLM-4.7-Flash \
  ...
```

Sweep wrappers (preferred harness):

- REAP: `$REAP/models/experimental/glm4_moe/experiments/prefill_opt_sweep/run_prefill_opt_sweep.sh`
- Flash: `$FLASH/models/experimental/glm4_moe_lite/experiments/prefill_opt_sweep/run_prefill_opt_sweep.sh`

---

## 3. Root causes already proven

### 3.1 REAP batched ≫ serial was MoE O(T²), not SDPA

- Batched path concatenates users → MoE sees `T = B·S`.
- Sparse matmul L1-bounds via `per_core_M ∝ num token-blocks` (`sparsity_block_size=32`). Without chunking, wall time scales ~**O(T²)** (not attention quadratic alone).
- Evidence (Jul 17 matrix, pre-chunk REAP, `batched_prefill_smoke/RESULTS.md`):

| ISL×B | T | batched_s | ≈t/T² (µs) |
|------:|--:|----------:|-----------:|
| 512×4 | 2048 | 173.9 | ~41 |
| 1024×4 | 4096 | 678.8 | ~40 |
| 2048×4 | 8192 | 2692 | ~40 |

- Flash already had PCM/chunking; REAP did not → REAP batched/serial ratios ~4–8×; Flash ~1.06–1.58×.

### 3.2 Host attn AR → fixed to `rs_ag`

- Prefill attention all-reduce was hardcoded **host** (PCIe D2H/H2D every layer) → dominated TTFT vs Llama-class numbers.
- **Landed:** default TG prefill attn reduce = **`rs_ag`** (`reduce_scatter` + `all_gather`). Override: `GLM4_MOE_ATTN_PREFILL_REDUCE_IMPL`.
- Sweep scripts **unset** `GLM4_MOE_ATTN_PREFILL_REDUCE_IMPL` so host is not forced.
- Logs confirm: `[ATTN PREFILL] all_reduce done (impl=rs_ag)`.
- Effect visible even without MoE chunking: Jul17 REAP 128×4 batched **22.3s** → Jul20 PCM0 baseline **12.7s** (same cell, rs_ag + current code).

### 3.3 Absolute TTFT vs Llama is not like-for-like

- Llama Galaxy docs (tens–hundreds of ms @ short ISL) are a different stack (dense/GQA, on-device CCL, different mesh/batching).
- REAP = 218B MoE, 92 layers, GQA+MoE; Flash = much smaller MLA lite stack.
- Do not use “match Llama TTFT” as a pass/fail gate for MoE GLM without matching workload and model class.

---

## 4. Code changes landed (uncommitted unless noted)

### REAP tree (`sdawle/glm47_reap_218b`)

Dirty (do **not** commit unless asked):

| File | What |
|------|------|
| `models/experimental/glm4_moe/tt/moe_tt.py` | Flash-style sparse prefill **PCM + CHUNK** loop |
| `models/experimental/glm4_moe/tt/attention_tt.py` | Prefill attn AR → **rs_ag** (env override) |
| `models/experimental/glm4_moe/tt/model_tt.py` | Batched multi-user prefill (`_prefill_batched`) |
| `models/experimental/glm4_moe/tt/decoder_layer_tt.py` | Prefill reduce / TIME_LAYER hooks |
| `models/experimental/glm4_moe/scripts/run_sweep_isl_batch.py` | Sweep harness tweaks |
| `models/experimental/glm4_moe/experiments/**` | Untracked experiment artifacts |

**REAP MoE knobs** (`moe_tt.py`) — **defaults baked 2026-07-21:**

| Env | Default when unset | Meaning |
|-----|--------------------|---------|
| `GLM4_MOE_MOE_SPARSE_PREFILL_PCM` | **adaptive:** `1` if `T > 256`, else `0` | Max sparsity blocks per sparse call; `0` disables PCM cap |
| `GLM4_MOE_MOE_SPARSE_PREFILL_CHUNK_TOKENS` | **adaptive:** `4096` if PCM>0, else `0` | Explicit chunk token budget |
| `GLM4_MOE_MOE_SPARSE_CHUNK_TOKENS` | same fallback if PREFILL_CHUNK unset | Alias for CHUNK |
| PCM=0 **and** CHUNK=0 | — | Legacy full-T path |
| Explicit env | always honored | Overrides adaptive policy |

`run_sweep_isl_batch.py` **forces** PCM=1 / CHUNK=4096 for production batched sweeps.

Also: `GLM4_MOE_BATCHED_PREFILL=1`, `GLM4_MOE_BATCHED_PREFILL_MAX_TOKENS=65536`, `GLM4_MOE_TIME_LAYER=1` for stage sync profiling.

### Flash tree (`sdawle/glm47_flash_wh_glx`)

- MoE PCM/chunking in `glm4_moe_lite/tt/moe_tt.py` — **default PCM baked to `2`** (2026-07-21).
- Dirty: `run_sweep_isl_batch.py` (sets PCM=2 / CHUNK=4096) + untracked `experiments/`.

**Flash MoE knobs:**

| Env | Default |
|-----|---------|
| `GLM4_MOE_LITE_MOE_SPARSE_PREFILL_PCM` | **`2`** (was `1`) |
| `GLM4_MOE_LITE_MOE_SPARSE_CHUNK_TOKENS` | `4096` (kept; PCM cap dominates) |
| `GLM4_MOE_LITE_BATCHED_PREFILL` | set by sweep |
| `GLM4_MOE_LITE_MOE_SPARSE_DISPATCH_IMPL` | `reduce` (also `a2a` / `all_to_all`) |

---

## 5. Measured results

### 5.1 Prior matrix (2026-07-17) — baseline before MoE chunk port

Full tables:
`/home/tt-admin/sdawle/glm47_reap_218b/tt-metal/models/experimental/glm4_moe/experiments/batched_prefill_smoke/RESULTS.md`
CSV: `.../batched_prefill_smoke/matrix_summary.csv` (28/28 ok).

Highlights:

- **REAP** 128×4: batched 22.3s vs serial 5.0s (**4.5×**); 512×8: 684 vs 89 (**7.6×**).
- **Flash** 512×8: batched 26.6 vs serial 16.9 (**1.58×**); 1024×16 almost parity (57.7 vs 54.3).

### 5.2 REAP smoke PCM0 vs PCM1 (2026-07-20) — DONE

Source:
`/home/tt-admin/sdawle/glm47_reap_218b/tt-metal/models/experimental/glm4_moe/experiments/prefill_opt_sweep/sweep_summary.csv`

| Tag | Mode | PCM | CHUNK | prefill_s | decode_ms |
|-----|------|----:|------:|----------:|----------:|
| reap_isl128_b4_batched_pcm1_chk4096 | batched | 1 | 4096 | **6.043** | 147.8 |
| reap_isl128_b4_serial_pcm1_chk4096 | serial | 1 | 4096 | 6.718 | 147.4 |
| reap_isl128_b4_batched_pcm0_chk0_baseline | batched | 0 | 0 | 12.710 | 147.8 |
| reap_isl128_b4_serial_pcm0_chk0_baseline | serial | 0 | 0 | **5.174** | 147.4 |

Takeaways:

- Chunking: batched **2.1×** vs PCM0 (6.0 vs 12.7); batched+PCM1 now **beats** serial+PCM1.
- Serial still best with **legacy PCM0** (5.17s); PCM=1 **hurts** small serial (~30%).

### 5.3 Flash PCM × CHUNK grid (2026-07-20) — DONE (all cells)

Source:
`/home/tt-admin/sdawle/glm47_flash_wh_glx/tt-metal/models/experimental/glm4_moe_lite/experiments/prefill_opt_sweep/sweep_summary.csv`

**Batched 512×8:**

| PCM | CHUNK | prefill_s |
|----:|------:|----------:|
| 1 | 4096 | **28.317** ← one-shot outlier (see §5.4) |
| 1 | 8192 | 15.333 |
| 2 | 4096 | 15.349 |
| 2 | 8192 | 15.296 |
| 4 | 4096 | 15.330 |
| 4 | 8192 | 15.260 |

**Batched 1024×16 / 2048×8 (CHUNK=4096):**

| Cell | PCM1 | PCM2 | PCM4 |
|------|-----:|-----:|-----:|
| 1024×16 | 59.764 | 57.183 | 57.186 |
| 2048×8 | 59.906 | 59.010 | 58.983 |

**Serial (complete):**

| Cell | PCM | CHUNK | prefill_s |
|------|----:|------:|----------:|
| 512×8 | 1 | 4096 | 16.951 |
| 512×8 | 2 | 4096 | 15.227 |
| 1024×16 | 1 | 4096 | 55.195 |
| 2048×8 | 1 | 4096 | 52.622 |

### 5.4 Short validation (2026-07-21) — DONE

Runner: `run_validation.sh`. Details: `VALIDATION.md` + both trees’ `validation_summary.csv`.

**REAP — clear speedup:**

| Cell | PCM/CHUNK | prefill_s | vs pair |
|------|-----------|----------:|--------:|
| 128×4 batched | 0/0 legacy | 22.370 | — |
| 128×4 batched | 1/4096 opt | **4.666** | **4.79×** |
| 512×8 batched | 0/0 legacy | **683.112** | — |
| 512×8 batched | 1/4096 opt | **24.317** | **28.1×** |
| 128×4 serial | 0/0 | **5.018** | best |
| 128×4 serial | 1/4096 | 5.569 | 1.11× slower |

**Flash — plateau confirmed; 28s cliff not reproduced:**

| Cell | PCM1/4096 | PCM2/4096 | Δ |
|------|----------:|----------:|---|
| 512×8 batched | 15.409 | 15.404 | ~0% |
| 1024×16 batched | 56.904 | 57.267 | noise |

---

## 6. Recommended defaults (post-validation) — **BAKED**

### REAP — production / sweep (**coded**)

- **Adaptive in `moe_tt.py`:** unset env → PCM=1 + CHUNK=4096 when `T > 256`, else PCM=0 + CHUNK=0.
- **Sweeps:** `run_sweep_isl_batch.py` sets PCM=1 / CHUNK=4096 explicitly.
- **Validated:** 4.8× @ 128×4 batched, **28×** @ 512×8 vs PCM0; serial 128×4 PCM1 ~11% slower (adaptive avoids this).
- **Do not** set `GLM4_MOE_ATTN_PREFILL_REDUCE_IMPL=host`.

### Flash — production / sweep (**coded**)

- **Default PCM=2** in `moe_tt.py` + sweep script; CHUNK stays 4096.
- Plateau ~15.3s @ 512×8; PCM1→PCM2 ~0% on validation; 28s PCM1/4096 = flaky outlier.
- At 1024×16 / 2048×8, PCM 2–4 is only a small / noisy gain vs PCM1; consistency still favors PCM≥2.

### Speed-up tables

See **`SPEEDUPS.md`** (this directory) for old vs new by ISL×B.

---

## 7. Where speedup comes from

| Lever | Impact | Notes |
|-------|--------|------|
| **MoE sparse PCM/chunk** | **Primary** for REAP batched (validated 28× @ 512×8); Flash already near plateau | Caps `per_core_M`; breaks O(T²). MoE is token-wise → chunk+concat correct. |
| **Attn `rs_ag`** | Large absolute win vs host AR | Already landed; keep. Not the Jul20 PCM delta (both arms use rs_ag). |
| **SDPA / attention compute** | Secondary for batched-vs-serial gap | Batched un-concats for RoPE/SDPA/per-slot fill; MoE stays flat on B·S. |
| **`paged_fill_cache` / `fill_cache`** | Per-user KV write; possible leftover cost | Batched still loops users for fill; further opt candidate. |
| **Decode** | Unchanged | REAP ~147ms, Flash ~79–87ms across cells. |

Attribution still thin: REAP `PHASE=profile` (`GLM4_MOE_TIME_LAYER=1`) **not run yet**.

---

## 8. Experiment artifacts & how to RESUME

### Paths

| What | Absolute path |
|------|----------------|
| REAP sweep dir | `/home/tt-admin/sdawle/glm47_reap_218b/tt-metal/models/experimental/glm4_moe/experiments/prefill_opt_sweep/` |
| REAP CSV / log / script | `sweep_summary.csv`, `validation_summary.csv`, `VALIDATION.md`, `SPEEDUPS.md`, `run_validation.sh`, `run_prefill_opt_sweep.sh`, `run_reap_full_isl_batch_sweep.sh` |
| REAP full-grid compare | `REAP_FULL_SWEEP_COMPARE.md`, `reap_full_new_summary.csv`, `old_baseline_inventory.csv`, `reap_full_*.png` |
| REAP per-cell logs | `reap_isl*_b*_*.log`, `val_reap_*.log`, `reap_full_isl*_b*_*.log` |
| Flash sweep dir | `/home/tt-admin/sdawle/glm47_flash_wh_glx/tt-metal/models/experimental/glm4_moe_lite/experiments/prefill_opt_sweep/` |
| Flash CSV / log / script | `sweep_summary.csv`, `validation_summary.csv`, `VALIDATION.md`, `run_prefill_opt_sweep.sh` |
| Jul17 matrix RESULTS | `/home/tt-admin/sdawle/glm47_reap_218b/tt-metal/models/experimental/glm4_moe/experiments/batched_prefill_smoke/RESULTS.md` |

### Resume REAP (smoke done; pcm + profile remaining)

```bash
cd /home/tt-admin/sdawle/glm47_reap_218b/tt-metal
# Wait until Flash idle (see §9)
RESUME=1 PHASE=pcm nohup \
  models/experimental/glm4_moe/experiments/prefill_opt_sweep/run_prefill_opt_sweep.sh \
  >> models/experimental/glm4_moe/experiments/prefill_opt_sweep/wrapper_pcm.log 2>&1 &

# After pcm:
RESUME=1 PHASE=profile nohup \
  models/experimental/glm4_moe/experiments/prefill_opt_sweep/run_prefill_opt_sweep.sh \
  >> models/experimental/glm4_moe/experiments/prefill_opt_sweep/wrapper_profile.log 2>&1 &
```

`RESUME=1` skips cells whose log already contains `prefill_s=`.
Phases: `smoke` | `pcm` | `profile` | `all`.

**REAP PHASE=pcm plan** (from script): PCM∈{0,1,2,4} @ 128×4 & 512×8 batched; PCM∈{1,2,4} @ 1024×8; serials; B=1 batched; one 1024×8 PCM0 baseline. Large cells are slow — budget hours.

### Resume Flash (if interrupted)

```bash
cd /home/tt-admin/sdawle/glm47_flash_wh_glx/tt-metal
RESUME=1 PHASE=all nohup \
  models/experimental/glm4_moe_lite/experiments/prefill_opt_sweep/run_prefill_opt_sweep.sh \
  >> models/experimental/glm4_moe_lite/experiments/prefill_opt_sweep/wrapper.log 2>&1 &
```

Phases: `pcm` | `batched_vs_serial` | `all`.

**Do not edit `run_*.sh` while it is running** (bash file-offset skew; see `batched_prefill_smoke/NOTES.md`).

---

## 9. Device exclusivity

**Never run Flash and REAP concurrent** on this Galaxy.

```bash
pgrep -af 'debug_run_full_tt_greedy|run_prefill_opt_sweep|run_sweep_isl_batch'
```

Scripts’ `wait_idle` matches only `python3 .*debug_run_full_tt_greedy.py` (and `run_sweep_isl_batch`).
**Do not kill** in-flight sweeps unless the user asks.

### Status at update (2026-07-21 ~04:43 UTC)

| Job | State |
|-----|--------|
| Flash sweep | **DONE** (all CSV cells) |
| Short validation (Flash→REAP) | **DONE** |
| REAP full ISL×B chunked sweep (20 cells) | **DONE** — 17 ok, 3 OOM @ T≥65536 |
| Device | **IDLE** after full sweep |

---

## 10. Status: DONE / IN PROGRESS / TODO

### DONE

- [x] Root-cause: REAP MoE O(T²) without chunking; host attn AR → rs_ag
- [x] REAP MoE PCM/CHUNK port in `moe_tt.py`
- [x] REAP smoke 128×4 PCM0 vs PCM1 (batched + serial)
- [x] Flash PCM×CHUNK batched grid (512×8, 1024×16, 2048×8) + serial cells
- [x] Jul17 full batched matrix RESULTS.md
- [x] **Short validation** (`VALIDATION.md`): REAP 128×4 + 512×8 batched speedup proven; Flash plateau re-checked; serial anti-pattern confirmed
- [x] **Adaptive / production defaults baked** (REAP adaptive T>256; Flash PCM=2; sweep scripts wired)
- [x] **`SPEEDUPS.md`** old vs new tables (existing numbers; unmeasured matrix cells noted)

### TODO for next agent (ranked)

1. Optional: `RESUME=1 PHASE=pcm` full REAP PCM grid (512×8 PCM2/4, 1024×8, etc.) — validation already proves the main claim.
2. Optional compact **new-path** grid for missing SPEEDUPS cells: batched PCM1 @ 128×{8,16,32}, 512×4, 1024×{4,8} (skip huge old re-runs).
3. `PHASE=profile` TIME_LAYER attribution (MoE vs attn vs fill_cache).
4. **Write `RESULTS.md`** in both `prefill_opt_sweep/` dirs (fold in VALIDATION + SPEEDUPS + sweep tables).
5. **Further opts (after knobs settled):**
   1. Confirm MoE vs attn vs fill_cache split via TIME_LAYER / Tracy
   2. Batched `paged_fill_cache` amortization / fewer per-user fills
   3. Flash `GLM4_MOE_LITE_MOE_SPARSE_DISPATCH_IMPL=a2a` experiment
   4. Re-check SDPA / un-concat overhead once MoE is linear
6. **Commit only if user asks** — keep rs_ag; do not reintroduce host attn AR.

---

## 11. Do not regress

- Keep **rs_ag** as TG prefill attn default; do not hardcode host AR.
- Sweep scripts must continue to **unset** `GLM4_MOE_ATTN_PREFILL_REDUCE_IMPL`.
- Do not run Flash+REAP concurrent.
- Do not kill running sweeps.
- **Do not commit** unless explicitly asked.
- Do not edit a running bash sweep script in place.

---

## Quick pickup checklist

1. `pgrep -af 'debug_run_full_tt_greedy|run_prefill_opt_sweep'` — who’s on device?
2. Read `SPEEDUPS.md` + `VALIDATION.md` + CSVs.
3. Optional: compact new-path grid for missing SPEEDUPS cells, or `PHASE=pcm` / `PHASE=profile`.
4. Draft RESULTS.md (fold SPEEDUPS + validation).
5. Canonical doc path: this HANDOFF.md.
