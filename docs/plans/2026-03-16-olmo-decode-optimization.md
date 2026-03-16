# OLMo Decode Optimization Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Close the decode performance gap between OLMo-3.1-32B (~17 tok/s) and Qwen3-32B (~3x faster) by moving QK-norm, MLP, and WO decode paths off 1-core DRAM onto L1 sharded multi-core.

**Architecture:** Four independent config/code changes in `olmo_model_config.py` and `llama_attention.py` / `llama_mlp.py`. No math changes. Each task is independently revertable. PCC checked after every task.

**Tech Stack:** TTNN, `olmo_model_config.py`, `llama_attention.py`, `llama_mlp.py`, existing PCC test infrastructure.

---

## PCC Gate (run after EVERY task)

```bash
export HF_MODEL=~/.cache/huggingface/hub/models--allenai--Olmo-3.1-32B-Think
export ARCH_NAME=wormhole_b0
export TT_METAL_HOME=$(pwd)
export PYTHONPATH=$(pwd)
source python_env/bin/activate

pytest models/demos/llama3_70b_galaxy/tests/test_olmo_e2e_pcc.py::TestOlmoE2EPCC::test_decode_pcc_1layer -xvs
pytest models/demos/llama3_70b_galaxy/tests/test_olmo_e2e_pcc.py::TestOlmoE2EPCC::test_decode_pcc_4layers -xvs
pytest models/demos/llama3_70b_galaxy/tests/test_olmo_e2e_pcc.py::TestOlmoE2EPCC::test_decode_pcc_64layers -xvs
```

**Gates — if any fail, revert the task and debug before continuing:**

| Test | Baseline | Gate |
|---|---|---|
| Decode 1L | 0.9983 | ≥ 0.998 |
| Decode 4L | 0.9963 | ≥ 0.995 |
| Decode 64L | 0.8165 | ≥ 0.80 |

**If a gate fails**, run per-op PCC to pinpoint the layer that regressed:
```bash
pytest models/demos/llama3_70b_galaxy/tests/test_olmo_e2e_pcc.py::TestOlmoE2EPCC::test_decode_per_op_pcc_4layers -xvs
```

---

## Background: Key Shapes & Dimensions

- `dim = 5120`, `dim_per_tp = 1280` (5120/4 col-devices)
- `n_q_heads = 40`, `n_local_heads = 5` (40/8 col-devices)
- `n_local_heads_padded = 8` (padded for ring matmul)
- `n_kv_heads = 8`, `n_local_kv_heads = 1` (8/8)
- `head_dim = 128`
- `batch_size = 32`, `batch_per_col_device = 8` (32/4 col-devices)
- `sub_core_grids`: cores (1,0)-(3,9) and (5,0)-(6,9) = 50 total
- `start_core = ttnn.CoreCoord(1, 0)`

### K-norm shape per device (decode)
- From `llama_rs_create_heads`: `[8-batch, 1-kv_head, 128-head_dim]` ROW_MAJOR L1_HEIGHT_SHARDED
- After tilize: `[8, 32, 128]` = 8 height-tiles × 4 width-tiles = 32 tiles total
- Optimal: **8 cores HEIGHT_SHARDED**, each core gets `[32, 128]` (1 batch item, full head_dim)

### Q-norm shape per device (decode)
- After slice (5 real heads) + reshape: `[8-batch, 640]` = `[8, 5×128]`
- After tilize (batch-pad 8→32): `[32, 640]` = 1 height-tile × 20 width-tiles
- Optimal: **20 cores WIDTH_SHARDED**, each core gets `[32, 32]` (1 tile/core)

---

## Task 1: K-norm L1 HEIGHT_SHARDED (8 cores)

**What this fixes:** K heads exit `llama_rs_create_heads` in L1 but are immediately moved to DRAM for norm ops (CSV rows 7-13: 31 us/layer wasted on DRAM). Fix: keep in L1 HEIGHT_SHARDED throughout the norm.

**Files:**
- Modify: `models/demos/llama3_70b_galaxy/tt/olmo_model_config.py` — add K-norm sharded configs after the `CREATE_HEAD_OUTPUT_MEMCFG` block (~line 660)
- Modify: `models/demos/llama3_70b_galaxy/tt/llama_attention.py:669-685` — use L1 sharded for K-norm

### Step 1: Add K-norm memory configs to `olmo_model_config.py`

Find the block ending with:
```python
        self.model_config["CREATE_HEAD_OUTPUT_MEMCFG"] = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                self.sub_core_grids,
                [32, 128],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        )
```

Insert immediately after:
```python
        # ==== Decode K-norm L1 Sharded Configs ====
        # K heads: [8-batch, 1-kv_head, 128-head_dim] → tiled [8, 32, 128]
        # HEIGHT shard: 8 cores, each gets [32, 128] (1 batch item, full head_dim for RMS norm)
        k_norm_n_cores = self.batch_per_col_device  # 8 (= batch_size // cluster_shape[1])
        k_norm_grid = ttnn.num_cores_to_corerangeset_in_subcoregrids(
            self.start_core, k_norm_n_cores, self.sub_core_grids, row_wise=True
        )
        self.model_config["OLMO_K_NORM_SHARDED_MEMCFG"] = ttnn.create_sharded_memory_config(
            [32, 128],
            core_grid=k_norm_grid,
            strategy=ttnn.ShardStrategy.HEIGHT,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )
        # Stats shape: [8, 32, 32] → each core gets [32, 32] (one tile of stats per batch item)
        self.model_config["OLMO_K_NORM_STATS_MEMCFG"] = ttnn.create_sharded_memory_config(
            [32, 32],
            core_grid=k_norm_grid,
            strategy=ttnn.ShardStrategy.HEIGHT,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )
        self.model_config["OLMO_K_NORM_SHARDED_PROGCFG"] = self.create_sharded_norm_config(k_norm_grid)
```

> **Note:** `self.batch_per_col_device` may not exist — use `self.max_batch_size // self.cluster_shape[1]` = `32 // 4` = `8` if needed. `self.create_sharded_norm_config` is defined in the base class and handles `LayerNormShardedMultiCoreProgramConfig` construction from a core grid; check `model_config.py` for its signature.

### Step 2: Update K-norm path in `llama_attention.py`

Find and replace the K-norm block (lines 666-685):

**Replace:**
```python
            # ---- K global norm: distributed rms_norm over 8 row devices (1024 total elements) ----
            # K [1,1,32,128] (1 KV head per device, 128 dims per row device)
            # all_gather on cluster_axis=0 (row axis, 8 devices) to compute global variance over 1024 dims
            k_heads_pre_rot_1BKD = ttnn.to_memory_config(k_heads_pre_rot_1BKD, ttnn.DRAM_MEMORY_CONFIG)
            k_heads_pre_rot_1BKD = ttnn.to_layout(k_heads_pre_rot_1BKD, ttnn.TILE_LAYOUT)
            k_stats = ttnn.rms_norm_pre_all_gather(
                k_heads_pre_rot_1BKD, dtype=ttnn.bfloat16, compute_kernel_config=self.compute_kernel_config_hifi2
            )
            k_stats_gathered = self._olmo_qk_norm_all_gather(k_stats, cluster_axis=0)
            ttnn.deallocate(k_stats)
            k_heads_pre_rot_1BKD = ttnn.rms_norm_post_all_gather(
                k_heads_pre_rot_1BKD,
                k_stats_gathered,
                epsilon=1e-6,
                weight=self.olmo_k_norm_weight,
                compute_kernel_config=self.compute_kernel_config_hifi2,
            )
            ttnn.deallocate(k_stats_gathered)
            k_heads_pre_rot_1BKD = ttnn.to_layout(k_heads_pre_rot_1BKD, ttnn.ROW_MAJOR_LAYOUT)
            k_heads_pre_rot_1BKD = ttnn.to_memory_config(k_heads_pre_rot_1BKD, k_mem_cfg)
```

**With:**
```python
            # ---- K global norm: L1 HEIGHT_SHARDED (8 cores), no DRAM roundtrip ----
            # K [8-batch, 1-kv_head, 128-head_dim]: stay in L1, shard to k_norm_grid
            k_heads_pre_rot_1BKD = ttnn.to_memory_config(
                k_heads_pre_rot_1BKD, self.model_config["OLMO_K_NORM_SHARDED_MEMCFG"]
            )
            k_heads_pre_rot_1BKD = ttnn.to_layout(
                k_heads_pre_rot_1BKD,
                ttnn.TILE_LAYOUT,
                memory_config=self.model_config["OLMO_K_NORM_SHARDED_MEMCFG"],
            )
            k_stats = ttnn.rms_norm_pre_all_gather(
                k_heads_pre_rot_1BKD,
                dtype=ttnn.bfloat16,
                compute_kernel_config=self.compute_kernel_config_hifi2,
                program_config=self.model_config["OLMO_K_NORM_SHARDED_PROGCFG"],
                memory_config=self.model_config["OLMO_K_NORM_STATS_MEMCFG"],
            )
            k_stats_gathered = self._olmo_qk_norm_all_gather(k_stats, cluster_axis=0)
            ttnn.deallocate(k_stats)
            k_heads_pre_rot_1BKD = ttnn.rms_norm_post_all_gather(
                k_heads_pre_rot_1BKD,
                k_stats_gathered,
                epsilon=1e-6,
                weight=self.olmo_k_norm_weight,
                compute_kernel_config=self.compute_kernel_config_hifi2,
                program_config=self.model_config["OLMO_K_NORM_SHARDED_PROGCFG"],
                memory_config=self.model_config["OLMO_K_NORM_SHARDED_MEMCFG"],
            )
            ttnn.deallocate(k_stats_gathered)
            k_heads_pre_rot_1BKD = ttnn.to_layout(k_heads_pre_rot_1BKD, ttnn.ROW_MAJOR_LAYOUT)
            k_heads_pre_rot_1BKD = ttnn.to_memory_config(k_heads_pre_rot_1BKD, k_mem_cfg)
```

> **Debugging note:** If `to_layout(..., memory_config=...)` is not supported, do `to_layout` then `to_memory_config` as two separate calls. If `OLMO_K_NORM_SHARDED_PROGCFG` causes a "grid mismatch" error, try passing `memory_config` only (no `program_config`) first to verify functional correctness, then add `program_config`.

### Step 3: Run PCC gate

```bash
export HF_MODEL=~/.cache/huggingface/hub/models--allenai--Olmo-3.1-32B-Think
export ARCH_NAME=wormhole_b0 && export TT_METAL_HOME=$(pwd) && export PYTHONPATH=$(pwd) && source python_env/bin/activate
pytest models/demos/llama3_70b_galaxy/tests/test_olmo_e2e_pcc.py::TestOlmoE2EPCC::test_decode_pcc_1layer -xvs
pytest models/demos/llama3_70b_galaxy/tests/test_olmo_e2e_pcc.py::TestOlmoE2EPCC::test_decode_pcc_4layers -xvs
pytest models/demos/llama3_70b_galaxy/tests/test_olmo_e2e_pcc.py::TestOlmoE2EPCC::test_decode_pcc_64layers -xvs
```

Expected: 1L ≥ 0.998, 4L ≥ 0.995, 64L ≥ 0.80. If any fail: revert this task.

### Step 4: Commit

```bash
git add models/demos/llama3_70b_galaxy/tt/olmo_model_config.py \
        models/demos/llama3_70b_galaxy/tt/llama_attention.py
git commit -m "perf(olmo): K-norm L1 HEIGHT_SHARDED (8 cores, remove DRAM roundtrip)"
```

---

## Task 2: Q-norm L1 WIDTH_SHARDED (20 cores)

**What this fixes:** Q-norm runs on 1 core DRAM (TilizeWithValPadding 47 us, LayerNormPre 13 us, LayerNormPost 24 us, Untilize 14 us = 98 us/layer). With 20-core WIDTH_SHARDED in L1 these drop to ~6 us total.

**Files:**
- Modify: `models/demos/llama3_70b_galaxy/tt/olmo_model_config.py` — add Q-norm sharded configs after the K-norm configs added in Task 1
- Modify: `models/demos/llama3_70b_galaxy/tt/llama_attention.py:691-734` — update Q-norm path

### Step 1: Add Q-norm memory configs to `olmo_model_config.py`

Insert immediately after the K-norm configs from Task 1:

```python
        # ==== Decode Q-norm L1 Sharded Configs ====
        # Q flat (after slice 5 real heads + reshape): [8-batch, 640] = [8, 5×128]
        # After tilize (pad batch 8→32): [32, 640] = 1 height-tile × 20 width-tiles
        # WIDTH shard: 20 cores, each gets [32, 32] (1 tile — full row stays split across 20 cores,
        # rms_norm_pre/post_all_gather handle distributed variance over all 20×8=160 chunks)
        q_norm_n_cores = self.n_local_heads * self.head_dim // self.tile_size  # 5 * 128 / 32 = 20
        q_norm_grid = ttnn.num_cores_to_corerangeset_in_subcoregrids(
            self.start_core, q_norm_n_cores, self.sub_core_grids, row_wise=True
        )
        self.model_config["OLMO_Q_NORM_SHARDED_MEMCFG"] = ttnn.create_sharded_memory_config(
            [32, 32],  # shard shape: 1 tile per core
            core_grid=q_norm_grid,
            strategy=ttnn.ShardStrategy.WIDTH,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )
        # Stats: [32, 32] per core (rms_norm_pre_all_gather output shape per core)
        self.model_config["OLMO_Q_NORM_STATS_MEMCFG"] = ttnn.create_sharded_memory_config(
            [32, 32],
            core_grid=q_norm_grid,
            strategy=ttnn.ShardStrategy.WIDTH,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )
        self.model_config["OLMO_Q_NORM_SHARDED_PROGCFG"] = self.create_sharded_norm_config(q_norm_grid)
```

### Step 2: Update Q-norm path in `llama_attention.py`

Find the Q-norm block starting at line 687 (`# ---- Q global norm`). Replace lines 691-734:

**Replace:**
```python
            q_heads_pre_rot_1BQD = ttnn.to_memory_config(q_heads_pre_rot_1BQD, ttnn.DRAM_MEMORY_CONFIG)
            q_batch = q_heads_pre_rot_1BQD.shape[1]  # batch dimension

            # Slice 5 real Q heads in dim 2: [1, batch, 8, 128] → [1, batch, 5, 128]
            q_real = ttnn.slice(q_heads_pre_rot_1BQD, [0, 0, 0, 0], [1, q_batch, self.n_local_heads, self.head_dim])
            ttnn.deallocate(q_heads_pre_rot_1BQD)

            # Reshape [1, batch, 5, 128] → [1, 1, batch, 640]: flatten heads into last dim
            q_flat = ttnn.reshape(q_real, [1, 1, q_batch, self.n_local_heads * self.head_dim])
            ttnn.deallocate(q_real)
            q_flat = ttnn.to_layout(q_flat, ttnn.TILE_LAYOUT)

            # Distributed global Q norm: all_gather on cluster_axis=0 (8 row devices)
            q_stats = ttnn.rms_norm_pre_all_gather(
                q_flat, dtype=ttnn.bfloat16, compute_kernel_config=self.compute_kernel_config_hifi2
            )
            q_stats_gathered = self._olmo_qk_norm_all_gather(q_stats, cluster_axis=0)
            ttnn.deallocate(q_stats)
            q_flat = ttnn.rms_norm_post_all_gather(
                q_flat,
                q_stats_gathered,
                epsilon=1e-6,
                weight=self.olmo_q_norm_weight_full_prefill,
                compute_kernel_config=self.compute_kernel_config_hifi2,
            )
            ttnn.deallocate(q_stats_gathered)
            q_flat = ttnn.to_layout(q_flat, ttnn.ROW_MAJOR_LAYOUT)

            # Undo reshape [1, 1, batch, 640] → [1, batch, 5, 128]
            q_real_normed = ttnn.reshape(q_flat, [1, q_batch, self.n_local_heads, self.head_dim])
            ttnn.deallocate(q_flat)

            # Pad 3 zero heads in dim 2: [1, batch, 5, 128] → [1, batch, 8, 128]
            n_pad = self.n_local_heads_padded - self.n_local_heads
            if n_pad > 0:
                q_heads_pre_rot_1BQD = ttnn.pad(q_real_normed, [(0, 0), (0, 0), (0, n_pad), (0, 0)], value=0.0)
                ttnn.deallocate(q_real_normed)
            else:
                q_heads_pre_rot_1BQD = q_real_normed
```

**With:**
```python
            q_heads_pre_rot_1BQD = ttnn.to_memory_config(q_heads_pre_rot_1BQD, ttnn.DRAM_MEMORY_CONFIG)
            q_batch = q_heads_pre_rot_1BQD.shape[1]  # batch dimension

            # Slice 5 real Q heads: [1, batch, 8, 128] → [1, batch, 5, 128]
            q_real = ttnn.slice(q_heads_pre_rot_1BQD, [0, 0, 0, 0], [1, q_batch, self.n_local_heads, self.head_dim])
            ttnn.deallocate(q_heads_pre_rot_1BQD)

            # Reshape [1, batch, 5, 128] → [1, 1, batch, 640]
            q_flat = ttnn.reshape(q_real, [1, 1, q_batch, self.n_local_heads * self.head_dim])
            ttnn.deallocate(q_real)

            # Move to L1 WIDTH_SHARDED (20 cores) before tilize to avoid 1-core DRAM tilize
            q_flat = ttnn.to_memory_config(q_flat, self.model_config["OLMO_Q_NORM_SHARDED_MEMCFG"])
            q_flat = ttnn.to_layout(
                q_flat,
                ttnn.TILE_LAYOUT,
                memory_config=self.model_config["OLMO_Q_NORM_SHARDED_MEMCFG"],
            )

            # Distributed global Q norm (20 cores L1 WIDTH_SHARDED)
            q_stats = ttnn.rms_norm_pre_all_gather(
                q_flat,
                dtype=ttnn.bfloat16,
                compute_kernel_config=self.compute_kernel_config_hifi2,
                program_config=self.model_config["OLMO_Q_NORM_SHARDED_PROGCFG"],
                memory_config=self.model_config["OLMO_Q_NORM_STATS_MEMCFG"],
            )
            q_stats_gathered = self._olmo_qk_norm_all_gather(q_stats, cluster_axis=0)
            ttnn.deallocate(q_stats)
            q_flat = ttnn.rms_norm_post_all_gather(
                q_flat,
                q_stats_gathered,
                epsilon=1e-6,
                weight=self.olmo_q_norm_weight_full_prefill,
                compute_kernel_config=self.compute_kernel_config_hifi2,
                program_config=self.model_config["OLMO_Q_NORM_SHARDED_PROGCFG"],
                memory_config=self.model_config["OLMO_Q_NORM_SHARDED_MEMCFG"],
            )
            ttnn.deallocate(q_stats_gathered)

            # Untilize in L1, move to DRAM for reshape
            q_flat = ttnn.to_layout(q_flat, ttnn.ROW_MAJOR_LAYOUT)
            q_flat = ttnn.to_memory_config(q_flat, ttnn.DRAM_MEMORY_CONFIG)

            # Undo reshape [1, 1, batch, 640] → [1, batch, 5, 128]
            q_real_normed = ttnn.reshape(q_flat, [1, q_batch, self.n_local_heads, self.head_dim])
            ttnn.deallocate(q_flat)

            # Pad 3 zero heads: [1, batch, 5, 128] → [1, batch, 8, 128]
            n_pad = self.n_local_heads_padded - self.n_local_heads
            if n_pad > 0:
                q_heads_pre_rot_1BQD = ttnn.pad(q_real_normed, [(0, 0), (0, 0), (0, n_pad), (0, 0)], value=0.0)
                ttnn.deallocate(q_real_normed)
            else:
                q_heads_pre_rot_1BQD = q_real_normed
```

> **Debugging note:** If `to_memory_config` into `OLMO_Q_NORM_SHARDED_MEMCFG` before `to_layout` fails (ROW_MAJOR sharded layout mismatch), try: `to_layout(TILE_LAYOUT)` first on DRAM, then `to_memory_config(OLMO_Q_NORM_SHARDED_MEMCFG)`. The 1-core TilizeWithValPadding is undesirable but not catastrophic — sharding the norm ops alone still saves ~50 us/layer.

### Step 3: Run PCC gate

```bash
pytest models/demos/llama3_70b_galaxy/tests/test_olmo_e2e_pcc.py::TestOlmoE2EPCC::test_decode_pcc_1layer -xvs
pytest models/demos/llama3_70b_galaxy/tests/test_olmo_e2e_pcc.py::TestOlmoE2EPCC::test_decode_pcc_4layers -xvs
pytest models/demos/llama3_70b_galaxy/tests/test_olmo_e2e_pcc.py::TestOlmoE2EPCC::test_decode_pcc_64layers -xvs
```

Expected: 1L ≥ 0.998, 4L ≥ 0.995, 64L ≥ 0.80. If any fail: revert this task.

### Step 4: Commit

```bash
git add models/demos/llama3_70b_galaxy/tt/olmo_model_config.py \
        models/demos/llama3_70b_galaxy/tt/llama_attention.py
git commit -m "perf(olmo): Q-norm L1 WIDTH_SHARDED (20 cores, 1-core DRAM bottleneck eliminated)"
```

---

## Task 3: MLP Decode L1 Path

**What this fixes:** Three OLMo-specific DRAM branches in `llama_mlp.py`:
1. SiLU-mul output forced to DRAM (`ff1ff3_mem_config = DRAM if is_olmo`) — should use `REDUCE_SCATTER_OUT_MEMCFG`
2. FF2 `line_all_gather` to DRAM — should use `FF2_IN_RING_MEMCFG` (L1 WIDTH_SHARDED)
3. FF2 uses `w2_interleaved` (no `program_config`, generic matmul) — should use `self.w2` + `FF2_DRAM_SHARDED_PROGCFG_OLMO`

**Files:**
- Modify: `models/demos/llama3_70b_galaxy/tt/llama_mlp.py:289-344`

### Step 1: Fix SiLU-mul memory config

Find (line ~289):
```python
        ff1ff3_mem_config = ttnn.DRAM_MEMORY_CONFIG if is_olmo else self.model_config["REDUCE_SCATTER_OUT_MEMCFG"]
```

Replace with:
```python
        ff1ff3_mem_config = self.model_config["REDUCE_SCATTER_OUT_MEMCFG"]
```

### Step 2: Fix FF2 all_gather input to L1

Find the OLMo FF2 `line_all_gather` (line ~303):
```python
        if is_olmo and mode == "decode":
            w2_in = self.tt_ccl.line_all_gather(
                ff1ff3,
                dim=3,
                cluster_axis=1,
                num_links=self.model_config["GALAXY_NUM_LINKS"],
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                buffer_key="BINARY_MUL",
                use_optimal_ccl_for_llama=True,
            )
```

Replace `memory_config=ttnn.DRAM_MEMORY_CONFIG` with:
```python
                memory_config=self.model_config["FF2_IN_RING_MEMCFG"],
```

### Step 3: Fix FF2 matmul to use DRAM-sharded weight + config

Find the OLMo FF2 matmul block (line ~324):
```python
        if is_olmo and mode == "decode":
            w2_out = ttnn.linear(
                w2_in,
                self.w2_interleaved,
                compute_kernel_config=self.args.compute_kernel_config_hifi2,
                dtype=ttnn.bfloat8_b,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            ttnn.deallocate(w2_in)
            self._debug_check_mlp("w2_out", w2_out)
            w2_out_sharded = ttnn.to_memory_config(w2_out, self.model_config["FF2_OUT_RING_MEMCFG_OLMO"])
            ttnn.deallocate(w2_out)
```

Replace with:
```python
        if is_olmo and mode == "decode":
            w2_out = ttnn.linear(
                w2_in,
                self.w2,
                compute_kernel_config=self.args.compute_kernel_config_hifi2,
                dtype=ttnn.bfloat8_b,
                program_config=self.model_config["FF2_DRAM_SHARDED_PROGCFG_OLMO"],
                memory_config=self.model_config["FF2_OUT_RING_MEMCFG_OLMO"],
            )
            ttnn.deallocate(w2_in)
            self._debug_check_mlp("w2_out", w2_out)
            w2_out_sharded = w2_out
```

> **Note:** `self.w2` is the DRAM-WIDTH_SHARDED weight used by the standard path. `FF2_DRAM_SHARDED_PROGCFG_OLMO` is defined in `olmo_model_config.py` at the `FF2_DRAM_SHARDED_PROGCFG_OLMO` key. `FF2_OUT_RING_MEMCFG_OLMO` is the L1 sharded output config for OLMo's 1280-wide output. If `self.w2` has a shape mismatch vs `w2_interleaved`, check how weights are loaded in `load_checkpoints.py` — `w2_interleaved` may have different padding.

### Step 4: Run PCC gate

```bash
pytest models/demos/llama3_70b_galaxy/tests/test_olmo_e2e_pcc.py::TestOlmoE2EPCC::test_decode_pcc_1layer -xvs
pytest models/demos/llama3_70b_galaxy/tests/test_olmo_e2e_pcc.py::TestOlmoE2EPCC::test_decode_pcc_4layers -xvs
pytest models/demos/llama3_70b_galaxy/tests/test_olmo_e2e_pcc.py::TestOlmoE2EPCC::test_decode_pcc_64layers -xvs
```

Expected: 1L ≥ 0.998, 4L ≥ 0.995, 64L ≥ 0.80. If any fail: revert step-by-step (Step 3 → Step 2 → Step 1) to isolate.

### Step 5: Commit

```bash
git add models/demos/llama3_70b_galaxy/tt/llama_mlp.py
git commit -m "perf(olmo): MLP decode to L1 (SiLU-mul, FF2 all_gather, FF2 matmul DRAM-sharded config)"
```

---

## Task 4: WO Ring Matmul Path

**What this fixes:** OLMo WO decode slices to 5 real heads, reassembles batch via `line_all_gather` (AllBroadcast, 1 core DRAM, 13 us) + Concat + Tilize (1 core DRAM, 8 us) before a DRAM matmul. The standard ring path avoids all this.

**Key prerequisite:** `wo_ring` weight must have zero rows for the 3 phantom heads (rows 640-1023 of K dim). Check this before any code change.

**Files:**
- Check: `models/demos/llama3_70b_galaxy/tt/load_checkpoints.py` — verify `wo_ring` loading pads K=640→1024 with zeros
- Modify: `models/demos/llama3_70b_galaxy/tt/llama_attention.py:958-1043` — replace OLMo WO path with standard ring path

### Step 1: Verify `wo_ring` weight loading

In `load_checkpoints.py`, search for where `wo_ring` / `wo` is loaded for OLMo. Verify that:
- Weight loaded has shape `(K=1024, N=1536)` per device (not `(640, 1280)`)
- Rows 640-1023 (phantom heads) are zero-filled

If `wo_ring` is loaded with `K=640` only (not padded), stop here — do not proceed with Task 4 until the weight loader is fixed to pad to `K=1024` with zeros. This is a blocker.

Run a quick check:
```python
# Add temporarily to load_checkpoints.py or run interactively:
import torch
wo = ...  # loaded wo_ring weight for device 0
print(f"wo_ring shape: {wo.shape}")  # expect (1024, 1536) or similar padded shape
print(f"Phantom rows nonzero: {wo[640:, :].abs().max()}")  # expect 0.0
```

### Step 2: Replace OLMo WO path in `llama_attention.py`

Find the OLMo-specific WO branch (line ~958, starting with the comment about SDPA→WO):

```python
            # OLMo SDPA→WO: slice padded heads, then all_gather batch across col devices.
            ...
            attn_output_cat = self.tt_ccl.line_all_gather(
                sdpa_flat,
                dim=2,
                cluster_axis=1,
                ...
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
```

This entire block (through `attn_output_cat = ttnn.to_layout(attn_output_cat, ttnn.TILE_LAYOUT)` for OLMo) should be replaced with the standard `all_gather_concat` path that Qwen3/Llama use.

**Replace the OLMo-specific block** (roughly lines 958-1013) with the standard path:
```python
            # Standard ring path: all_gather_concat gathers batch across col devices
            # OLMo: 8-padded-heads output (3 phantom heads are zero from Q-norm pad)
            # wo_ring weight has zero rows for phantom heads → correct WO output
            attn_output_cat = self.tt_ccl.all_gather_concat(
                attn_output_1G4D_sharded,
                dim=1,
                cluster_axis=1,
                num_links=self.model_config["GALAXY_NUM_LINKS"],
                memory_config=self.model_config["SHARDED_ATTN_WO_INPUT_RING_MEMCFG"],
                num_heads=decode_num_heads,
            )
            ttnn.deallocate(attn_output_1G4D_sharded)
```

Then in the WO matmul block (~line 1015), remove the `if self.is_olmo:` branch and use the standard path for OLMo too (the `wo_ring` + `WO_DECODE_RING_PROGCFG` path).

> **Note:** `attn_output_1G4D_sharded` is the variable name used just before the OLMo-specific branch. Check the exact variable name at that point in the code. `decode_num_heads = self.n_local_heads_padded` (8 padded heads for OLMo). If `SHARDED_ATTN_WO_INPUT_RING_MEMCFG` has a shape mismatch with the 8-head OLMo output, check its shard shape — it should be `[32, 1024//RING_SIZE]` for OLMo (K=1024 padded heads).

### Step 3: Run PCC gate

```bash
pytest models/demos/llama3_70b_galaxy/tests/test_olmo_e2e_pcc.py::TestOlmoE2EPCC::test_decode_pcc_1layer -xvs
pytest models/demos/llama3_70b_galaxy/tests/test_olmo_e2e_pcc.py::TestOlmoE2EPCC::test_decode_pcc_4layers -xvs
pytest models/demos/llama3_70b_galaxy/tests/test_olmo_e2e_pcc.py::TestOlmoE2EPCC::test_decode_pcc_64layers -xvs
```

Expected: 1L ≥ 0.998, 4L ≥ 0.995, 64L ≥ 0.80.

### Step 4: Commit

```bash
git add models/demos/llama3_70b_galaxy/tt/llama_attention.py
git commit -m "perf(olmo): WO decode ring matmul path (skip AllBroadcast/Concat/Tilize on 1 core)"
```

---

## Task 5: Measure Performance + Update BRINGUP_LOG

### Step 1: Run E2E demo (64 layers, traced)

```bash
export HF_MODEL=~/.cache/huggingface/hub/models--allenai--Olmo-3.1-32B-Think
export ARCH_NAME=wormhole_b0 && export TT_METAL_HOME=$(pwd) && export PYTHONPATH=$(pwd) && source python_env/bin/activate
pytest models/demos/llama3_70b_galaxy/demo/demo_olmo_decode.py::test_olmo_demo -v -k "quick"
```

Record:
- Decode speed: `ms/tok` and `tok/s/user`
- Compare against baseline: 55.5 ms/tok @ 18.0 tok/s/user

Expected: ~40-47 ms/tok → ~21-25 tok/s/user (~25-45% improvement).

### Step 2: Update BRINGUP_LOG.md

Add a new session entry:
```markdown
## Session: 2026-03-16 Decode Optimization

### Status: COMPLETE
### Changes
- K-norm: L1 HEIGHT_SHARDED (8 cores), removed DRAM roundtrip
- Q-norm: L1 WIDTH_SHARDED (20 cores), eliminated 1-core DRAM bottleneck
- MLP decode: SiLU-mul + FF2 all_gather → L1, FF2 matmul → DRAM-sharded config
- WO decode: ring matmul path (removed AllBroadcast/Concat/Tilize)

### PCC Results (post-optimization)
| Test | PCC | Gate | Status |
|------|-----|------|--------|
| Decode 1L | [fill] | ≥ 0.998 | [PASS/FAIL] |
| Decode 4L | [fill] | ≥ 0.995 | [PASS/FAIL] |
| Decode 64L | [fill] | ≥ 0.80 | [PASS/FAIL] |

### Performance Results
| Metric | Before | After | Delta |
|--------|--------|-------|-------|
| Decode ms/tok | 55.5 | [fill] | [fill] |
| tok/s/user | 18.0 | [fill] | [fill] |
```

### Step 3: Commit

```bash
git add models/demos/llama3_70b_galaxy/BRINGUP_LOG.md docs/plans/2026-03-16-olmo-decode-optimization-design.md docs/plans/2026-03-16-olmo-decode-optimization.md
git commit -m "docs(olmo): decode optimization results and design docs"
```

---

## Rollback Plan

Each task is independently revertable. Order of revert priority (if issues arise):
1. **Task 4 (WO)**: Highest risk. If PCC fails, revert immediately — `wo_ring` phantom-head padding must be verified first.
2. **Task 3 (MLP)**: Medium risk. Revert sub-steps 3→2→1 to isolate (`w2` vs `w2_interleaved` is the main uncertainty).
3. **Task 2 (Q-norm)**: Low risk. If sharded progcfg fails, try memory_config only first.
4. **Task 1 (K-norm)**: Lowest risk — same core count as current, just moves to L1.
