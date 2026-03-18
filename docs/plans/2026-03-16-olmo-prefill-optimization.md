# OLMo Prefill Config Optimization Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Close the ~4x prefill performance gap between OLMo and Qwen3 by fixing suboptimal core grids and missing configs in `olmo_model_config.py`.

**Architecture:** OLMo-3.1-32B uses the same TG (4x8 mesh) infrastructure as Qwen3-32B. The performance gap is caused by conservative core grids chosen during initial bring-up — SDPA uses 28 cores vs Qwen3's 70, and matmul grids are similarly under-provisioned. All changes are config-only in `olmo_model_config.py`; no attention/MLP/decoder code changes needed.

**Tech Stack:** TTNN, `olmo_model_config.py`, existing PCC test infrastructure.

---

## Context

### Key OLMo Dimensions (per device)
- `dim_per_tp = 1280` (5120 / 4)
- `n_local_q_heads = 5` (40 / 8 col devices)
- `n_local_kv_heads = 1` (8 / 8 col devices)
- `qkv_size_per_device_prefill = 896` (28 tiles)
- `wo_k_prefill = 640` (20 tiles), `wo_n = 1280` (40 tiles)
- `intermediate_dim_per_tp = 3456` (108 tiles)

### Current vs Target Core Grids

| Config | OLMo Current | Qwen3 (target) | Speedup Estimate |
|--------|-------------|----------------|-----------------|
| **SDPA prefill** | (7, 4) = 28 | **(7, 10) = 70** | ~2.5x |
| **XQKV prefill** | (4, 10) = 40 | **(7, 10) = 70** | ~1.5x |
| **WO prefill** | (5, 10) = 50 | **(7, 10) = 70** | ~1.3x |
| **WO short seq** | (4, 10) = 40 | **(7, 10) = 70** | ~1.5x |

### File to Modify
- `models/demos/llama3_70b_galaxy/tt/olmo_model_config.py`

### PCC Gate (run after EVERY task)

Every task must end with this PCC check. If it fails, revert the task and debug before continuing.

```bash
export HF_MODEL=~/.cache/huggingface/hub/models--allenai--Olmo-3.1-32B-Think
export ARCH_NAME=wormhole_b0
export TT_METAL_HOME=$(pwd)
export PYTHONPATH=$(pwd)
source python_env/bin/activate

pytest models/demos/llama3_70b_galaxy/tests/test_olmo_e2e_pcc.py::TestOlmoE2EPCC::test_prefill_pcc_1layer -xvs
```

**Baseline PCC values** (from BRINGUP_LOG.md — must match or exceed after each change):

| Test | Metric | Baseline | Gate |
|------|--------|----------|------|
| Prefill 1L | Hidden state PCC | 0.9998 | ≥ 0.999 |
| Prefill 1L | Logits PCC | 0.9992 | ≥ 0.999 |
| Prefill 64L | Hidden state PCC | 0.9776 | ≥ 0.97 |
| Prefill 64L | Token match | ✓ (4815) | Must match |
| Decode 1L | PCC | 0.9983 | ≥ 0.998 |

If any PCC drops below the gate, **revert** the change immediately.

---

### Task 0: Remove KV cache zeroing from profiled path

Profiler rows 49-50 show `BinaryNgDeviceOperation` on the full KV cache `4096x64x128` at ~324 us each. For 64 layers, that's 128 ops × 324 us = **~41.5 ms** of wasted time per prefill. These come from `ttnn.mul(cache, 0)` calls in the demo code that zero the cache between warmup/trace steps. With paged attention, this is unnecessary — each user writes to their own pages via the page table, so old data is never accessed.

**Files:**
- Modify: `models/demos/llama3_70b_galaxy/demo/demo_olmo_decode.py:293-297,315-319`

**Step 1: Remove cache zeroing between warmup and trace capture**

In `demo_olmo_decode.py`, delete lines 293-297:

```python
    # Reset KV cache after warmup to avoid stale data from compile run
    for layer in tt_model.layers:
        k_cache, v_cache = layer.attention.layer_past
        k_cache = ttnn.mul(k_cache, 0, output_tensor=k_cache)
        v_cache = ttnn.mul(v_cache, 0, output_tensor=v_cache)
```

And delete lines 315-319 (same pattern, "Reset KV cache after trace capture").

The paged attention page table ensures each user's prefill writes to the correct pages. Stale data in other pages is never read because SDPA only reads positions up to the current sequence length.

**Step 2: Run PCC gate**

```bash
pytest models/demos/llama3_70b_galaxy/tests/test_olmo_e2e_pcc.py::TestOlmoE2EPCC::test_prefill_pcc_1layer -xvs
```

Expected: PASS, PCC matches baseline (≥ 0.999). Cache zeroing removal should not affect PCC since paged attention isolates user data.

---

### Task 1: Increase SDPA prefill grid from (7,4) to (7,10)

This is the single highest-impact change. SDPA is O(seq² × head_dim) and dominates prefill compute. Going from 28→70 cores should yield ~2-2.5x speedup on SDPA alone.

**Files:**
- Modify: `models/demos/llama3_70b_galaxy/tt/olmo_model_config.py:949-971`

**Step 1: Edit the SDPA config**

Replace the `sdpa_progcfg` function (lines 949-971) with:

```python
        def sdpa_progcfg(seq_len):
            if seq_len <= 256:
                return ttnn.SDPAProgramConfig(
                    compute_with_storage_grid_size=[7, 10],
                    exp_approx_mode=False,
                    q_chunk_size=64,
                    k_chunk_size=64,
                )
            elif seq_len <= 2048:
                return ttnn.SDPAProgramConfig(
                    compute_with_storage_grid_size=[7, 10],
                    exp_approx_mode=False,
                    q_chunk_size=128,
                    k_chunk_size=128,
                )
            else:
                return ttnn.SDPAProgramConfig(
                    compute_with_storage_grid_size=[7, 10],
                    exp_approx_mode=False,
                    q_chunk_size=256,
                    k_chunk_size=256,
                )
```

Changes:
- Grid `[7, 4]` → `[7, 10]` (28→70 cores, matching Qwen3)
- Added `exp_approx_mode=False` (matching Qwen3 for accuracy)

**Step 2: Run prefill PCC test (1 layer)**

```bash
pytest models/demos/llama3_70b_galaxy/tests/test_olmo_e2e_pcc.py::TestOlmoE2EPCC::test_prefill_pcc_1layer -xvs
```

Expected: PASS with PCC ≥ 0.999 (same as before — grid change should not affect accuracy).

If this **crashes with L1 OOM**: Fall back to `[7, 8]` (56 cores) or `[7, 6]` (42 cores) and retest.

---

### Task 2: Increase XQKV prefill grid from (4,10) to (7,10)

The QKV projection matmul currently uses only 40 cores. Qwen3 uses 70 cores for the same operation.

**Files:**
- Modify: `models/demos/llama3_70b_galaxy/tt/olmo_model_config.py:1023-1050`

**Step 1: Edit the XQKV config**

Replace lines 1023-1050 with:

```python
        # QKV prefill config (uses unpadded QKV weights: 5 Q heads, N=896 per device)
        # 896 / 32 = 28 tiles. Use 7 cores for N (28/7=4 tiles/core), matching Qwen3 grid
        qkv_pf_n_tiles = qkv_size_per_device_prefill // 32  # 28 tiles
        qkv_pf_n_cores = 7  # 28 / 7 = 4 evenly
        qkv_pf_per_core_n = qkv_pf_n_tiles // qkv_pf_n_cores  # 4
        self.model_config["XQKV_PREFILL_PROGCFG"] = (
            lambda seq_len: self.matmul_1d_config(
                seq_len,
                self.dim // 4,
                qkv_size_per_device_prefill,
                grid=ttnn.CoreGrid(x=7, y=10),
                overwrite_per_core_k=8,
            )
            if seq_len == 128
            else (
                ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
                    compute_with_storage_grid_size=(qkv_pf_n_cores, 10),
                    in0_block_w=8,
                    out_subblock_h=1,
                    out_subblock_w=2,
                    per_core_M=max(1, 8 if seq_len >= 2048 else seq_len // self.tile_size // 8),
                    per_core_N=qkv_pf_per_core_n,
                    transpose_mcast=False,
                    fused_activation=None,
                    fuse_batch=seq_len <= 2048,
                )
            )
        )
```

Changes:
- `qkv_pf_n_cores`: 4 → 7 (28 tiles / 7 = 4 tiles/core, divides evenly)
- `matmul_1d_config` grid: `(x=4, y=10)` → `(x=7, y=10)` for seq_len=128
- `compute_with_storage_grid_size`: `(4, 10)` → `(7, 10)` for general case
- `out_subblock_w`: 1 → 2 (4 tiles per core, 4%2=0 ✓, matches Qwen3 pattern)

**Step 2: Run prefill PCC test (1 layer)**

```bash
pytest models/demos/llama3_70b_galaxy/tests/test_olmo_e2e_pcc.py::TestOlmoE2EPCC::test_prefill_pcc_1layer -xvs
```

Expected: PASS with PCC ≥ 0.999.

---

### Task 3: Increase WO prefill grid from (4-5,10) to (7,10)

The output projection matmul uses 40-50 cores. Qwen3 uses 70.

**Files:**
- Modify: `models/demos/llama3_70b_galaxy/tt/olmo_model_config.py:1052-1075`

**Step 1: Edit the WO config**

Replace lines 1052-1075 with:

```python
        # WO prefill config (uses unpadded WO: K=640, N=1280)
        # 640/32 = 20 tiles for K, 1280/32 = 40 tiles for N
        # Use 7 cores for N: ceil(40/7) = 6 tiles/core (matching Qwen3 grid)
        wo_n_tiles = self.dim // 4 // 32  # 40 tiles
        wo_n_cores = 7
        wo_per_core_n = math.ceil(wo_n_tiles / wo_n_cores)  # ceil(40/7) = 6
        self.model_config["WO_PREFILL_PROGCFG"] = (
            lambda seq_len: self.matmul_1d_config(
                seq_len, wo_k_prefill, self.dim // 4, grid=ttnn.CoreGrid(x=7, y=10), overwrite_per_core_k=8
            )
            if seq_len == 128
            else (
                ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
                    compute_with_storage_grid_size=(wo_n_cores, 10),
                    in0_block_w=8,
                    out_subblock_h=1,
                    out_subblock_w=2,
                    per_core_M=max(1, 4 if seq_len >= 1024 else seq_len // self.tile_size // 8),
                    per_core_N=wo_per_core_n,
                    transpose_mcast=False,
                    fused_activation=None,
                    fuse_batch=seq_len <= 1024,
                )
            )
        )
```

Changes:
- `wo_n_cores`: 5 → 7
- `wo_per_core_n`: 8 → `ceil(40/7)` = 6 (multicast handles non-even distribution)
- `matmul_1d_config` grid: `(x=4, y=10)` → `(x=7, y=10)` for seq_len=128
- `compute_with_storage_grid_size`: `(5, 10)` → `(7, 10)`
- `in0_block_w`: 4 → 8 (matching Qwen3 — more K tiles per block, better compute density)
- `overwrite_per_core_k`: 4 → 8 (matching Qwen3)
- `out_subblock_w`: stays 2 (6%2=0 ✓)
- `per_core_M` threshold: changed to match Qwen3's `seq_len >= 1024` with value 4
- `fuse_batch` threshold: changed to `<= 1024` (matching Qwen3)

**Step 2: Run prefill PCC test (1 layer)**

```bash
pytest models/demos/llama3_70b_galaxy/tests/test_olmo_e2e_pcc.py::TestOlmoE2EPCC::test_prefill_pcc_1layer -xvs
```

Expected: PASS with PCC ≥ 0.999.

---

### Task 4: Add missing XQKV_PREFILL_MINIMAL_PROGCFG and KV_PREFILL_MEM_CFG

These configs are defined in Qwen3 and the base class but missing from OLMo. They're needed for seq_len >= 4096 when the minimal matmul path is used, and for KV cache sharding during prefill.

**Files:**
- Modify: `models/demos/llama3_70b_galaxy/tt/olmo_model_config.py` — add after the XQKV_PREFILL_PROGCFG block (~line 1050)

**Step 1: Add the missing configs**

Insert after the `XQKV_PREFILL_PROGCFG` assignment (after the new Task 2 code), before the WO config:

```python
        def prefill_xqkv_minimal_matmul_config(seq_len):
            if seq_len <= 128:
                return ttnn.MinimalMatmulConfig(
                    M_block_size=8,
                    K_block_size=8,
                    N_block_size=8,
                    subblock_h=4,
                    subblock_w=2,
                    compute_with_storage_grid_size=ttnn.CoreCoord(7, 7),
                )
            elif seq_len <= 1024:
                return ttnn.MinimalMatmulConfig(
                    M_block_size=8,
                    K_block_size=8,
                    N_block_size=8,
                    subblock_h=4,
                    subblock_w=2,
                    compute_with_storage_grid_size=ttnn.CoreCoord(7, 8),
                )
            else:
                return ttnn.MinimalMatmulConfig(
                    M_block_size=8,
                    K_block_size=8,
                    N_block_size=8,
                    subblock_h=1,
                    subblock_w=8,
                    compute_with_storage_grid_size=ttnn.CoreCoord(7, 8),
                )

        self.model_config["XQKV_PREFILL_MINIMAL_PROGCFG"] = prefill_xqkv_minimal_matmul_config

        self.model_config["KV_PREFILL_MEM_CFG"] = lambda seq_len: ttnn.create_sharded_memory_config(
            (((self.n_kv_heads // self.cluster_shape[1]) * seq_len // (8 * 8)), self.head_dim),
            ttnn.CoreGrid(y=8, x=8),
            ttnn.ShardStrategy.HEIGHT,
            ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )
```

These are copied from the Qwen3/base configs (identical structure). The XQKV minimal is currently referenced only in commented-out code, but KV_PREFILL_MEM_CFG may be needed by the base attention code for longer sequences.

**Step 2: Run PCC gate**

```bash
pytest models/demos/llama3_70b_galaxy/tests/test_olmo_e2e_pcc.py::TestOlmoE2EPCC::test_prefill_pcc_1layer -xvs
```

Expected: PASS, PCC matches baseline (≥ 0.999). These are additive configs that don't affect the current code path.

---

### Task 5: Full PCC validation (all tests)

After all config changes (Tasks 0-4), run the comprehensive PCC suite to confirm no regressions.

**Step 1: Run 1-layer prefill PCC**

```bash
pytest models/demos/llama3_70b_galaxy/tests/test_olmo_e2e_pcc.py::TestOlmoE2EPCC::test_prefill_pcc_1layer -xvs
```

Expected: PASS, hidden state PCC ≥ 0.999, logits PCC ≥ 0.999.

**Step 2: Run 64-layer prefill PCC**

```bash
pytest models/demos/llama3_70b_galaxy/tests/test_olmo_e2e_pcc.py::TestOlmoE2EPCC::test_prefill_pcc -xvs
```

Expected: PASS, hidden state PCC ≥ 0.97, token match ✓ (4815).

**Step 3: Run 1-layer decode PCC (regression check)**

```bash
pytest models/demos/llama3_70b_galaxy/tests/test_olmo_e2e_pcc.py::TestOlmoE2EPCC::test_decode_pcc_1layer -xvs
```

Expected: PASS, PCC ≥ 0.998. Decode configs are NOT changed, but confirms no accidental side effects.

**Step 4: Run E2E demo (output quality check)**

```bash
pytest models/demos/llama3_70b_galaxy/demo/demo_olmo_decode.py -v -k "single"
```

Expected: Coherent output (same quality as baseline). This also validates the KV cache zeroing removal from Task 0 doesn't corrupt multi-user inference.

---

### Task 6: Measure performance improvement

Run the E2E demo and record TTFT (time-to-first-token) for prefill comparison.

**Step 1: Run E2E demo**

```bash
pytest models/demos/llama3_70b_galaxy/demo/demo_olmo_decode.py -v -k "single"
```

Record: TTFT (ms), decode speed (tok/s).

**Step 2: Compare against baseline**

Baseline from BRINGUP_LOG.md:
- TTFT: 266 ms (128 token prefill)
- Decode: 55.5 ms/tok @ 18.0 tok/s/user

Expected improvement: TTFT should drop by 2-3x (from SDPA + matmul grid improvements). Decode should be unchanged (decode configs not modified).

**Step 3: If TTFT improvement is < 2x**

Run profiler to identify remaining bottlenecks:
```bash
python scripts/run_profiler_sweep.sh
```

Analyze output to determine if QK-norm all-gathers or other operations are now the dominant bottleneck. This would motivate Approach B (fusing QK-norm all-gathers).

---

### Task 7: Commit

```bash
git add models/demos/llama3_70b_galaxy/tt/olmo_model_config.py models/demos/llama3_70b_galaxy/demo/demo_olmo_decode.py
git commit -m "perf(olmo): optimize prefill core grids and remove KV cache zeroing

- SDPA grid: (7,4)→(7,10) — 28→70 cores, ~2.5x on SDPA
- XQKV grid: (4,10)→(7,10) — 40→70 cores
- WO grid: (5,10)→(7,10) — 50→70 cores
- Added missing XQKV_PREFILL_MINIMAL_PROGCFG and KV_PREFILL_MEM_CFG
- Removed unnecessary KV cache zeroing between prefill iterations (~41ms saved)"
```

---

## Rollback Plan

If any task causes crashes or PCC regression:
1. Each config change is independent — revert just the failing config
2. For SDPA: try (7, 8) → (7, 6) → (7, 5) as progressively smaller grids
3. For matmuls: revert to original core counts; the N-dimension divisibility is the main constraint
4. For missing configs: these are additive and shouldn't break anything existing
