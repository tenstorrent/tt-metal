# BH 130-core Grid Migration Design Doc (research-only)

Single comprehensive design document for migrating
`models/demos/qwen3_6_galaxy_v2/` from the inherited Wormhole-style
60-core `sub_core_grids` to the full Blackhole Galaxy 130-core Tensix
grid. Research-only — no code modified, no device opened. Hand-off
material for the V2-grid implementation pass.

Baseline references throughout the doc:
- V2-2 commit `45b2138d759` — current `sub_core_grids` definition site
- V2-tracy-2 (post V2-14 + V2-16) — decode device-time attribution
  (PERF.md §1109 onwards)
- V2-16 real-loop baseline — 62.10 ms/step → 16.10 tok/s/user (64L
  traced decode, PERF.md:1006 / 1058 / 1484)
- V2-17c (in flight) — recurrent kernel multi-core + readout fusion;
  uses `grid="auto"` and is therefore separable from this work

---

## 1. BH actual grid dimensions

### 1.1 Authoritative documented dims

Two consistent statements inside the v2 tree document the live mesh
grid sizes:

- `tt/prefetcher_common.py:61` — “Determine actual compute grid from
  device; WH TG: (7,10), BH GLX: (13,10)”
- `tt/qwen_model_config.py:424` — “Use the actual device compute
  grid (WH: 7x10, BH: 13x10).”

So on a real BH P150 Galaxy mesh:

```
mesh_device.compute_with_storage_grid_size() => CoreCoord(x=13, y=10)
```

That is **130 Tensix worker cores per chip** (13 columns × 10 rows),
vs Wormhole TG’s 70 (7 × 10). The user’s “130 cores per chip” count
matches `13 × 10` exactly.

The same file shows the *worker* sub-grid layout that the BH P150
prefetcher (when enabled) reserves around. Both WH and BH carve out
sender columns at `col 0` and `col 4`; the BH-specific branch in
`prefetcher_common.py:95-101` declares the worker `SubDevice` as:

```python
ttnn.CoreRange(ttnn.CoreCoord(1, 0), ttnn.CoreCoord(3, grid_size.y - 1)),  # cols 1-3
ttnn.CoreRange(ttnn.CoreCoord(5, 0), ttnn.CoreCoord(grid_size.x - 1, grid_size.y - 1)),  # cols 5-12
```

= 3 cols (1-3) + 8 cols (5-12) = 11 cols × 10 rows = **110 worker
cores** with prefetcher on, **120 worker cores** with prefetcher off
(col 4 reclaimable).

### 1.2 What qwen36 currently uses

`tt/qwen36_model_config.py:188-192` (V2-2 / commit `45b2138d759`)
hard-codes:

```python
self.sub_core_grids = ttnn.CoreRangeSet(
    [
        ttnn.CoreRange(ttnn.CoreCoord(1, 0), ttnn.CoreCoord(6, 9)),
    ]
)
```

= 6 cols (1-6) × 10 rows = **60 cores**.

This was derived from the inherited WH 70-core grid (cols 1-6 × rows
0-9) minus the col-4 prefetcher carve-out reclaimed (50 → 60). It
does not know about BH’s extra 6 columns (cols 7-12) — those 60
cores are dark today for every op that consumes `self.sub_core_grids`.

**Net unused compute area on BH: 130 − 60 = 70 cores idle** (54 % of
the chip) for every model-config-driven op.

Note: this is wider than the headline “sub_core_grids unused” count
because the `start_core = (1, 0)` dispatcher reservation makes col 0
unavailable. Even on a worker-only grid the usable maximum is
120 cores (cols 1-12 × rows 0-9 with col 4 dropped), not 130. The
realistic V2-grid target is **120 cores**, not 130; the doc uses
that number from this point on.

### 1.3 Methodology to re-confirm on a live mesh

Since this doc was produced read-only with another agent on the
device, the source-code claim above (`13 × 10`) is what we trust.
A one-off device check before landing any code changes:

```python
# python_env (3.10) on a real BH GLX mesh, no model imports:
import ttnn
mesh = ttnn.open_mesh_device(ttnn.MeshShape(8, 4))
grid = mesh.compute_with_storage_grid_size()
print(grid)  # expected: CoreCoord(x=13, y=10)
print(mesh.dram_grid_size())  # for the DRAM CRS used by weight uploads
ttnn.close_mesh_device(mesh)
```

If `grid != (13, 10)`, every grid-derived per-op program config in
this doc must be re-derived. (E.g. BH P100 SKUs report a different
shape per the WH=12 / BH-P150=8 / BH-P100=7 DRAM-bank note in
`tt/qwen_model_config.py:1816`.) Open question — see §5.

---

## 2. Enumerate sites that consume `sub_core_grids`

All concrete v2 consumers of `self.sub_core_grids` or the hard-coded
`(1,0)→(6,9)` region. File-paths under
`/home/tt-admin/ssinghal/tt-metal/models/demos/qwen3_6_galaxy_v2/`.

### 2.1 Definition site

| file | line | role |
|---|---|---|
| `tt/qwen36_model_config.py` | 188-192 | qwen3.6 v2 `sub_core_grids` = (1,0)→(6,9), 60 cores |
| `tt/qwen36_model_config.py` | 194-198 | `sub_core_grid_topk` = (1,0)→(3,9), 30 cores |
| `tt/qwen36_model_config.py` | 199 | `start_core = CoreCoord(1, 0)` |
| `tt/qwen_model_config.py` | 209-220 | inherited Qwen3-32B `sub_core_grids` = cols 1-3 + 5-6 (50 cores, prefetcher-aware split) |

### 2.2 In-config consumers (per-op program / memory configs)

Inside `tt/qwen36_model_config.py::_populate_program_configs`:

| line | model_config key | what it does | grid migration delta |
|---|---|---|---|
| 402-409 | `PAGED_SDPA_DECODE_PROGCFG` | paged SDPA decode — 48-core sub-grid carved from `sub_core_grids` | re-derive 48-core layout inside the 120-core grid; `compute_with_storage_grid_size=(8, 6)` is the bounding rectangle and must change |
| 411-418 | `SDPA_DECODE_PROGCFG` | non-paged SDPA decode — 32-core sub-grid | re-derive 32-core layout; `compute_with_storage_grid_size=(8, 4)` is bounding rect; can stay if 32 worker cores still fit one rectangle |
| 432-434 | `SCORES_BATCHED_MM_OUTPUT_MEMCFG` | post-SDPA score memcfg — batch-size sub-grid | unchanged (uses `batch_size_per_device_group` cores, well below 120) |
| 484-489 | `RS_CREATE_HEADS_PACKET_WORKER_CRS` | RS+create-heads packet worker buffer — hard-coded 5 cores (cols 1-3 / rows 0-1) | unchanged (already a 5-core packet worker, no need to scale) |
| 602-604 | `attn_input_sub_core_grid` (SHARDED_ATTN_INPUT_MEMCFG) | 32-core band carved from `sub_core_grids` row-wise | can stay 32-core (driven by `dram_shard_core_grid_for_k(dim)`); width fine to leave |
| 631-633 | shard for `CREATE_HEAD_INPUT_MEMCFG` | 10-core 5×2 band (cols 1-2) | unchanged (10 cores hard-coded; no scaling benefit) |
| 638-642 | `CREATE_HEAD_INPUT_MEMCFG` | uses the 10-core CRS above | unchanged |
| 648-655 | `CREATE_HEAD_OUTPUT_MEMCFG` | uses **entire** `sub_core_grids` as shard CRS, shape (32, 128) | shard CRS grows from 60 → 120 cores; total shard volume = 60 × (32 × 128) = 245 KB → 120 × (32 × 128) = 491 KB → still tiny vs L1, fine |
| 663-665 | `GATHER_USERS_MEMCFG` | 32-core band carved row-wise | unchanged (32 cores) |
| 674-679 | `PACKET_WORKER_CRS` for RS interim | hard-coded 8 cores (cols 1-3 / rows 1-3) | unchanged |
| 688-690 | `FF1_CRS_RS_OUT` | 30-core band carved row-wise | could grow to 60-core band (`30 → 60`) to halve per-core RS-out shard width on the 5120-wide TG output — but RS topology constraints may apply (see §5) |
| 704-705 | `SHARDED_NORM_ATTN_PRGM_CFG`, `SHARDED_NORM_MLP_PRGM_CFG` | sharded norm program configs over `attn_input_grid`, `mlp_core_grid` | grids come from `dram_shard_core_grid_for_k(dim)` which uses DRAM core count, NOT `sub_core_grids` — already auto-scales with `mesh_device.dram_grid_size()` |
| 745-756 | LM-head ring CRS (`LM_HEAD_*_GRID`) | hard-coded coordinate lists from `tt/model_config.py:70-176` (32-core ring + 16-core input + 24-core output, all in cols 1-6) | doesn’t use `sub_core_grids` directly — coords are absolute. To use cols 7-12 you must extend the `LM_HEAD_32_GRID` / `LM_HEAD_INPUT_GRID` / `LM_HEAD_OUTPUT_GRID` lists (32→64 ring), but the ring count also affects `LM_HEAD_RING_SIZE=24` ↔ `LM_HEAD_TG_RING_PROGCFG` shape math (§3) |

Ring-input / output via `PREFETCHER_NOC1_GRID` (24 coords inside cols
1-6): same caveat — coordinates are absolute (cols 1, 2, 5, 6 only).
Consumers at lines 346, 350 of `qwen36_model_config.py` and at
`tt/model_config.py:1337`. Migration would require either (a)
extending the ring to 32/48 cores using cols 7-12 or (b) keeping
the ring small and using the new cores only for the SDPA / norm
slots that already auto-scale.

### 2.3 In-block consumers (TT-NN block code)

| file | line(s) | what it does |
|---|---|---|
| `tt/llama_ccl.py` | 44 | `sub_device_crs = args.sub_core_grids` (decode mode sub-device worker set) |
| `tt/llama_rope.py` | 108, 166-179, 253-255 | RoPE picks up `args.sub_core_grids` for trans-mat core grid; legacy path falls back to the inherited 50-core split |
| `tt/llama_attention.py` | reads `model_config["*"]` keys above (none direct on sub_core_grids) | indirect — picks up SDPA/QKV/WO progcfgs and memcfgs from `model_config` |
| `tt/llama_mlp.py` | reads `model_config["*"]` keys above (none direct on sub_core_grids) | indirect — picks up FF1/FF2/FF3 ring progcfgs and memcfgs from `model_config` |
| `tt/lm_head.py` | 113-114, 152, 155, 165 | reads `model_config["LM_HEAD_TG_RING_PROGCFG"]` + `LM_HEAD_OUT_RING_MEMCFG` + `SHARDED_LM_HEAD_INPUT_32_RING_MEMCFG` (all defined in qwen36_model_config.py from `LM_HEAD_*_GRID` constants) |
| `tt/qwen36_delta_attention.py` | 1000, 1465-1467 | hard-codes kernel-launch CoreRangeSets at `(2, 5)` (V2-18 single core) and `(2,5)→(5,5)` (V2-17c 4-core) — **inside the current 60-core grid; not on the new BH cols 7-12**. Will continue to work after grid expansion |
| `tt/qwen36_delta_attention.py` | 998 comment, 1462 comment | doc-only comments referring to “cols 1-6 × rows 0-9”; should be updated when the underlying grid changes |
| `tt/distributed_norm.py` | – | uses `model_config["GATHER_USERS_MEMCFG"]` etc.; no direct sub_core_grids reference |

### 2.4 Out-of-block consumers (demos / tests)

These hard-code an explicit `sub_core_grids` ttnn.CoreRangeSet rather
than reading `args.sub_core_grids`. Each will need a matching update
when the model-side grid changes:

| file | line(s) | role |
|---|---|---|
| `demo/demo_qwen_decode.py` | 238-, 270, 273, 302, 305 | sampling / plus_one / manual_seed all hard-code the 60-core sub_core_grids |
| `demo/demo_performance.py` | 210-, 239, 242, 248, 273, 276, 281 | performance demo |
| `demo/demo_decode.py` | 224, 228, 316, 319, 376, 381, 384, 414, 417 | reference decode demo (these reference `model_args.sub_core_grids` — auto-updates) |
| `tests/test_qwen_accuracy.py` | 266, 302, 307 | accuracy test (hard-coded 60-core) |
| `tests/test_llama_accuracy.py` | 210, 246, 251 | legacy accuracy test (hard-coded 50-core split) |
| `tests/unit_tests/test_sampling.py` | 292, 346 | uses `model_args.sub_core_grids` (auto-updates) |
| `tests/test_decode_perf_intrace.py` | 420, 425 | hard-coded 1-core CRS for sampling |
| `tests/test_ccl_buffer_keys.py` | 28-29 | **unit test mocks `sub_core_grids.num_cores() == 60`** — must update if num_cores changes |
| `tests/module_tests/test_qwen_qk_norm.py` | 54 | hard-coded test grid (50 cores, comment says “50 cores total”) |
| `tests/unit_tests/test_llama_attention.py` | 121-123, 186-188, 258 | uses `model_args.sub_core_grids` (auto-updates) |
| `tests/test_llama_model.py` | 209, 213, 274, 277, 338 | uses `model_args.sub_core_grids` (auto-updates) |
| `tests/unit_tests/test_llama_ops.py` | 160-, 187, 209, 279-, 311, 334, 352-, 371, 385, 393-, 415, 419 | parametrized over multiple sub_core_grids variants (still valid) |
| `tests/test_device_construct_smoke.py` | 105 | smoke prints the value, no constraint |

### 2.5 Inherited base-class consumers

`tt/qwen_model_config.py` (the inherited Qwen3-32B config that
qwen3.6 v2 derives from indirectly via `TtModelArgs`) consumes
`sub_core_grids` at 48, 82, 110, 914-915, 925-926, 970, 1180, 1188,
1317, 1463. Qwen3.6 v2 **does NOT call into this class’s
`_populate_program_configs`** — it overrides with its own
`_populate_program_configs` at `qwen36_model_config.py:271`. The
inherited site does not need to change for V2-grid.

---

## 3. Per-op program config implications

For every op category, what must change at `qwen36_model_config.py`
when `sub_core_grids` grows from 60 → 120 cores.

### 3.1 Sharded norm program configs (`SHARDED_NORM_*_PRGM_CFG`)

These come from `create_sharded_norm_config(grid)` (defined in the
inherited base, `tt/model_config.py:2676`). The grid is computed by
`dram_shard_core_grid_for_k(dim)` (`tt/model_config.py:2372`) which
uses the **DRAM bank count**, not `sub_core_grids`. So the norms
already auto-scale with the device DRAM grid (12 banks on WH, 8 on
BH P150, 7 on BH P100; per `tt/qwen_model_config.py:1816, 1833`).

**Migration delta: NONE in qwen36_model_config — but verify
`dram_shard_core_grid_for_k(5120)` produces a grid whose worker
cores actually live inside the new `sub_core_grids` region.** If
not, the norm’s WIDTH-shard placement may collide with the
prefetcher carve-out (col 4).

### 3.2 Matmul-1d-ring program configs

Four ops: `XQKV_DECODE_RING_PROGCFG`, `WO_DECODE_RING_PROGCFG`,
`FF1_3_TG_RING_PROGCFG`, `FF2_TG_RING_PROGCFG`. All built by
`matmul_1d_ring_config()` (`tt/model_config.py:2473`) over the
24-core `PREFETCHER_NOC1_GRID`. The ring topology is **independent
of the model-config grid** — the cores are hard-listed in
`tt/model_config.py:43-68`. Those 24 coords live in cols 1, 2, 5, 6
of the WH grid (also valid on BH).

Migration options:

1. **Leave the 24-core ring intact, only widen the sub-grids for the
   memcfg consumers below.** Wins: zero risk to the most-tuned op
   (`matmul_1d_ring_config` shape math is hard-tied to `RING_SIZE
   = 24` at line 344, 736 with explicit alignment to padded widths
   `dim_padded_24_cores = 6144`, `qkvg_per_col_padded_24_cores =
   3840`, `intermediate_dim_per_tp_padded_24_cores = 3840`).
2. **Extend the ring to 32 or 48 cores by adding cols 7-12 entries
   to a new `PREFETCHER_NOC1_GRID_BH`.** Requires recomputing the
   padded widths (e.g. 32-core ring → tile-alignment = 32 × 32 =
   1024, ff_n_padded would shift from 3840 to a multiple of 1024;
   probably 4096). All per-core shard widths in
   `SHARDED_QKV_OUT_RING_MEMCFG` / `SHARDED_FF12_OUT_RING_MEMCFG` /
   etc. drop by 24 / 32 = ×0.75. The ring topology constraint
   `num_blocks_total == num_cores` (`tt/model_config.py:2494`) must
   still hold.

The biggest matmul wins from option 2: per-core in0/out tile counts
fall, which directly cuts the per-call kernel time on the
`MatmulDeviceOperation` and `MinimalMatmulDeviceOperation` slices.
Option 1 keeps the same per-call kernel time but frees the 60 idle
cores for SDPA / norm / sampling ops — smaller absolute win.

### 3.3 LM-head ring config

`LM_HEAD_TG_RING_PROGCFG` is built from `matmul_1d_ring_lm_head_config()`
over a hand-rolled 32-core ring (`LM_HEAD_32_GRID`, 24-core
output ring `LM_HEAD_OUTPUT_GRID`, 16-core input
`LM_HEAD_16_GRID`). Per-col vocab = 62208 (line 738), tile-aligned
to `LM_HEAD_RING_SIZE * tile_size = 768`. Output shard width
`per_col_vocab_padded // 24 = 2592` (line 776), reshard width
`per_col_vocab_padded // 32 = 1944` (line 783). To extend, every
shape math is tied to the literal `LM_HEAD_RING_SIZE = 24`. A 32 or
48 core ring would require:

- new tile-alignment 32 × 32 = 1024 or 48 × 32 = 1536
- new per-col vocab padding: `ceil(62208 / 1024) × 1024 = 63488`
  (32-core) or `ceil(62208 / 1536) × 1536 = 63488` (48-core)
- new `LM_HEAD_32_GRID` / `LM_HEAD_OUTPUT_GRID` coordinate lists
  extending into cols 7-12

LM head is **19.2 % of decode device time** per V2-tracy-2
(PERF.md:1214). A 24 → 48 core ring at 2× width per shard halves
the per-core load → ~2× per-call latency improvement on the lm_head
matmul. Per-step delta: 19.2 % × 62.10 ms = 11.9 ms × 0.5 = **~6 ms
saved** if the 2× scales linearly. Realistic 30-40 % of theoretical
= 1.5-2.5 ms saved on the lm_head alone.

### 3.4 CCL persistent buffers (`QKV_BF16`, `WO_AG_BF16`, …)

The bf16 persistent CCL buffers used by line-all-reduce paths (V2-14)
are allocated in `tt/llama_ccl.py` based on `sub_core_grids` (line 44
`sub_device_crs`). The buffer shapes are tied to the
`REDUCE_SCATTER_OUT_MEMCFG` shard CRS (`FF1_CRS_RS_OUT` at line 688
= 30-core band). Growing the RS-out band to 60 cores halves the
per-core shard width on the 5120-wide output (5120 / 30 = 170.67 →
not tile-aligned → padded; vs 5120 / 60 = 85.33 → also not
tile-aligned). Practical viable counts: **40-core band** (5120 / 40
= 128 = 4 tiles, clean tile-aligned).

Buffer L1 footprint per chip: `(32 × shard_width)` bytes × bf16 =
`32 × 128 × 2 = 8 KB / core` for a 40-core band, total `40 × 8 KB =
320 KB`. Still trivial vs BH chip L1 (~1.5 MB usable per core).

### 3.5 SDPA program configs (`SDPA_DECODE_PROGCFG`, `PAGED_SDPA_DECODE_PROGCFG`)

`SDPA_DECODE_PROGCFG` already uses a 32-core sub-CRS of
`sub_core_grids` (line 411-414). `PAGED_SDPA_DECODE_PROGCFG` uses
48-core (line 401-409). On the new 120-core grid, both can either:

- (a) stay at 32 / 48 cores (drop-in, no shape change) and just
  pick a different sub-rectangle of cols 7-12, or
- (b) grow to 64-96 cores. SDPA scales nearly linearly with core
  count on small batch sizes (1-8 users / device), so option (b)
  gives a direct per-call latency cut.

Critical: `SDPA_PROGCFG` and `SDPA_PROGCFG_FLEXIBLE_CHUNK` (lines
373, 391) already use the **full device grid** `(grid.x, grid.y)` —
on BH that is `(13, 10)` = 130 cores. **SDPA prefill already
benefits from BH-130 dims today**; the WH→BH grid migration is
mostly about widening the *decode-side* sub-grids that the SDPA
decode config explicitly caps.

### 3.6 Sampling / plus_one / argmax

Sampling ops in `demo/*` and `tests/*` hard-code `sub_core_grids`
explicitly rather than reading `args.sub_core_grids`. These need
matching updates (§2.4). The argmax / topk paths use
`sub_core_grid_topk` (line 194) — a 30-core 3-col band — which is
narrower than `sub_core_grids` and can stay as-is.

### 3.7 RoPE trans-mat core grid

`llama_rope.py:253-255` builds the trans-mat core grid by carving
`batch_size_per_device_group` cores from `args.sub_core_grids`. On
the 120-core grid the carve-out succeeds trivially. No further
change needed; this is auto-driven from `args.sub_core_grids`.

---

## 4. Migration order (PCC + coherency gated)

Single multi-step task, ordered to gate each grid widening against a
working baseline.

### Step 0: live-mesh verification (one-off, no model open)

```python
# python_env on real BH GLX mesh, agent must NOT collide with current device user.
import ttnn
mesh = ttnn.open_mesh_device(ttnn.MeshShape(8, 4))
print(mesh.compute_with_storage_grid_size())  # expect CoreCoord(13, 10)
print(mesh.dram_grid_size())                  # expect CoreCoord(8, 1) on BH P150
ttnn.close_mesh_device(mesh)
```

If the assert fails (e.g. P100 SKU returning `(13, 10)` but different
DRAM count), abort and re-derive §3.1 (`dram_shard_core_grid_for_k`).

### Step 1: widen `sub_core_grids` only

Modify `tt/qwen36_model_config.py:188-192` to:

```python
self.sub_core_grids = ttnn.CoreRangeSet(
    [
        ttnn.CoreRange(ttnn.CoreCoord(1, 0), ttnn.CoreCoord(3, 9)),   # 30 cores, cols 1-3
        ttnn.CoreRange(ttnn.CoreCoord(5, 0), ttnn.CoreCoord(12, 9)),  # 80 cores, cols 5-12
    ]
)
# 110 cores total (worker-grid with prefetcher carve-out)
```

OR (if prefetcher truly off and col 4 reclaimable):

```python
self.sub_core_grids = ttnn.CoreRangeSet(
    [
        ttnn.CoreRange(ttnn.CoreCoord(1, 0), ttnn.CoreCoord(12, 9)),  # 120 cores, cols 1-12
    ]
)
```

The qwen3.6 v2 docstring (`qwen36_model_config.py:16-22`) already
says “prefetcher is auto-disabled in `TtModelArgs` (incompatible
8-DRAM-bank topology), but the existing config still carves col 4
out of `sub_core_grids` as if it were a prefetcher sender column.
With no prefetcher running, col 4 is unused — reclaim it.” Same
logic extends to cols 7-12.

**Gate**: existing 4-layer PCC tests
(`tests/test_4layer_hybrid_pcc.py`,
`tests/test_decode_eager_per_layer_pcc.py`,
`tests/test_decode_eager_64L_pcc.py`) must all still pass at PCC ≥
0.99. Decode coherency check
(`tests/test_decode_64L_real_prompt_pcc.py`) must still produce
matching tokens. No tracy needed yet — this step doesn’t change op
shapes, only widens the CRS that ops can pick from.

Expected device-time delta after step 1 alone: **0**. The ops that
read `sub_core_grids` directly (`CREATE_HEAD_OUTPUT_MEMCFG` at
line 648-655, `llama_ccl.sub_device_crs` at line 44, RoPE trans-mat
at `llama_rope.py:253`) will pick more cores, but the per-op shape
math doesn’t use them yet.

### Step 2: widen SDPA decode

Change `SDPA_DECODE_PROGCFG` from 32 → 64 cores and
`PAGED_SDPA_DECODE_PROGCFG` from 48 → 96 cores (lines 401-414). The
bounding rectangles `compute_with_storage_grid_size=(8, 6)` /
`(8, 4)` need re-derivation to span the new sub-grids.

**Gate**: 4-layer PCC test for one full-attention layer
(`tests/test_layer3_full_attention_forward_pcc.py`). Tracy capture
of 4-step decode — confirm `SDPAOperation` per-call avg drops from
16.70 µs (V2-tracy-2:1225) toward ~8 µs.

### Step 3: widen `FF1_CRS_RS_OUT` band

`FF1_CRS_RS_OUT` (line 688) goes from 30 → 40 cores. Pad to
multiple of `40 × 32 = 1280`; `dim = 5120 % 1280 == 0` (cleanly 4
tiles per core). `REDUCE_SCATTER_OUT_MEMCFG` shape changes from
`(32, 32)` → `(32, 32)` (still 1 tile / core but on 40 cores
instead of 30).

**Gate**: MLP PCC test
(`tests/unit_tests/test_llama_attention.py` — runs CCL+matmul), full
4L hybrid PCC.

### Step 4: widen LM-head ring (highest-leverage single op)

Extend `LM_HEAD_32_GRID` to 48 cores (add 16 entries in cols 7-12)
and bump `LM_HEAD_RING_SIZE` from 24 → 48 at line 736. Re-derive
the per-col vocab padding (line 740-742) and all shard widths
(`LM_HEAD_OUT_RING_MEMCFG`, `LM_HEAD_OUT_RING_RESHARD_MEMCFG`).

**Gate**: end-to-end coherency test on 64L decode
(`tests/test_decode_64L_real_prompt_pcc.py` produces matching tokens
vs reference at the 32-token horizon). LM head is the most logits-
visible op, so a coherency drift here will surface as wrong tokens
immediately.

### Step 5: optional — widen FF / WO / XQKV ring to 48-core

Riskier — see §3.2 option 2. Tile-alignment math forces
recomputing `qkvg_per_col_padded`, `intermediate_dim_per_tp_padded`,
and the 24-core-derived constants at lines 170-172. Recommend
**deferring** this until steps 1-4 are landed and tracy confirms
how much matmul time remains.

### Step 6: tracy V2-tracy-3 sheet

Re-run `demo/tracy_perf_4L_4T.py` with V2-14 + V2-16 flags on and
the new grid. Capture:
- `MatmulDeviceOperation` per-step (was 27.6 ms aggregate / step at
  V2-tracy-2)
- `MinimalMatmulDeviceOperation` per-step (was 21.7 ms aggregate /
  step at V2-tracy-2)
- `SDPAOperation` per-step (was 0.54 ms aggregate / step)
- Decode total per-step (was 112.6 ms aggregate / step → 3.52 ms /
  chip / step)
- Real-loop traced ms/step + tok/s/user

Compare against the 62.10 ms / 16.10 tok/s/user baseline.

---

## 5. Risk register

### 5.1 DST register count limits

Tensix BH has the same DST register bank topology as WH but more of
them per chip. Per-core DST count is unchanged. **Risk: NONE for
expanded core count.** The per-core matmul block size math
(`out_block_h × out_block_w × dtype_bytes ≤ DST_size`) stays
identical because each core works on the same per-shard volume.

### 5.2 Shard volume bounds

The CREATE_HEAD_OUTPUT shard volume on the new grid:
- 60 cores × (32 × 128) × bf16 = 60 × 32 × 128 × 2 = 491 KB across
  the cluster, ~8 KB / core (tiny; well under the BH per-core L1
  budget of ~1.5 MB).
- 120 cores × (32 × 128) × bf16 = 983 KB across, 8 KB / core (same
  per-core; doubles aggregate L1 across the chip).

**Risk: NONE.** All sharded memcfgs stay tile-sized at the per-core
level; widening only adds more cores, not more L1 per core.

### 5.3 Ring topology requirements

`matmul_1d_ring_config` asserts `num_blocks_total == num_cores`
(`tt/model_config.py:2494`). This means `M / out_block_h × N /
out_block_w == num_cores` must hold exactly. When widening
`RING_SIZE` from 24 → 32 / 48, all five padded widths
(`dim_padded_24_cores`, `qkvg_per_col_padded_24_cores`,
`intermediate_dim_per_tp_padded_24_cores`, plus the LM-head
widths) must be recomputed to satisfy the constraint or the assert
fires at construction time.

**Risk: HIGH for step 5 only.** Mitigation: keep ring size 24 and
only widen the *non-ring* consumers (steps 1-4).

### 5.4 L1 footprint check

V2-13 / V2-14 introduced persistent L1 buffers for residual,
DeltaNet conv state, RoPE decode pos tensor. Current allocations
are bounded by `args.sub_core_grids.num_cores() × per-shard`. On
the 120-core grid these would double:

- Persistent residual buffer (V2-14): `(32, dim_per_tp // num_cores)
  × bf16` per chip. dim_per_tp = 1280. 60 cores → 32 × ~22 × 2 =
  ~1.4 KB / core. 120 cores → 32 × ~11 × 2 = ~0.7 KB / core.
  Aggregate is *unchanged* (doubling cores halves per-core).
- DeltaNet conv state (V2-14 / 17b.7): residency is in DRAM, not
  L1-shard; unaffected.
- KV cache: lives in DRAM; sharding shape is driven by `n_kv_heads`
  not `sub_core_grids`. Unaffected.

**Risk: LOW.** All scaling is per-shard-per-core, which is
bounded by aggregate / num_cores — no exponential growth.

### 5.5 Per-op precision risk (V2-11 D/F mem-config swap family)

The V2-11 precision cliff was triggered by an interleaved↔sharded
mem-config swap on the WO output (PERF.md §340-380). The V2-grid
migration changes the **placement** of shards (which cores) but
keeps the **shape** of shards (rows × cols per core). However:

- LM head step 4 changes the per-shard width
  (`per_col_vocab_padded / RING_SIZE`). This is shape-changing →
  same family of risk as V2-11.
- Step 5 (matmul ring widening) changes `qkvg_per_col_padded`
  values → also shape-changing.
- Steps 1-3 are placement-only.

**Mitigation: gate every shape-changing step with a per-block PCC
test, NOT just an end-to-end coherency test.** Per the CLAUDE.md
debug protocol, isolate any failing op with a unit-level test
before claiming the step is done.

### 5.6 BH SKU variance

`tt/qwen_model_config.py:1816` documents WH=12 / BH P150=8 / BH
P100=7 DRAM banks. Step 0’s live-mesh check should confirm the
target SKU before any grid widening lands — a P100 mesh has a
different DRAM banking that may affect the
`dram_shard_core_grid_for_k` results used in §3.1.

### 5.7 Inherited base class

`tt/qwen_model_config.py:209-214` still defines the **prefetcher-
aware 50-core split**. Qwen3.6 v2 doesn’t use it, but anything that
inadvertently inherits and re-derives from `TtQwenModelArgs.__init__`
(not `TtQwen36ModelArgs.__init__`) will pick up the old grid.
Search for any module that constructs `TtQwenModelArgs` directly —
the v2 tree doesn’t, but a regression test added later might.

### 5.8 Test mocks

`tests/test_ccl_buffer_keys.py:28-29` hard-codes
`sub_core_grids.num_cores.return_value = 60`. Update this mock
before/at the same time as the model-config widening, otherwise the
test will silently pass against stale assumptions.

---

## 6. Expected perf gain

### 6.1 Back-of-envelope

Per V2-tracy-2 (PERF.md:1213-1239 / 1247-1249), the decode device
time slices are:

| op | aggregate / step | per chip / step | share |
|---|---:|---:|---:|
| MatmulDeviceOperation | 27.6 ms | 0.86 ms | 24.5 % |
| MinimalMatmulDeviceOperation (lm_head) | 21.7 ms | 0.68 ms | 19.2 % |
| ReduceScatter | 21.0 ms | 0.66 ms | 18.7 % |
| AllGatherAsync | 17.2 ms | 0.54 ms | 15.3 % |
| AllGather | 11.8 ms | 0.37 ms | 10.4 % |
| BinaryNg | 4.1 ms | 0.13 ms | 3.6 % |
| other | … | … | 8.3 % |
| **decode total** | **112.6 ms** | **3.52 ms** | **100 %** |

Matmul (`Matmul + MinimalMatmul`) is **43.8 % of decode device
time**. Theoretical maximum benefit from doubling compute area (60
→ 120 cores) on those ops: 43.8 % × 50 % = **21.9 % of decode device
time**.

But:
- LM head's `MinimalMatmul` uses a hand-rolled 24-core ring and
  needs explicit ring re-derivation (§3.3) to actually benefit.
- The four `Matmul` ops (XQKV / WO / FF1_3 / FF2) also use the
  24-core ring — same caveat (§3.2).
- The CCL slice (44.4 %) does NOT benefit from more compute cores;
  it’s bandwidth-bound, not compute-bound.

**Realistic device-time delta if we land steps 1-4 (NOT step 5):**
- LM head 24 → 48 ring: ~30 % per-call improvement on the 19.2 %
  slice → **3.7 % of decode device time saved** = ~4.2 ms aggregate
  / step → ~0.13 ms / chip / step.
- SDPA 32 → 64 cores: ~30 % per-call improvement on the 0.5 % slice
  → negligible (~0.1 % saved).
- Norm + create-heads slot widening: <0.5 % saved.
- **Total step 1-4 device-time saving: ~4.3 ms aggregate / step =
  ~0.13 ms / chip / step.**

**Realistic device-time delta if step 5 lands (matmul ring 24 → 48):**
- Matmul 24.5 % share × 30 % per-call = **7.4 % decode device-time
  saved** = ~8.3 ms aggregate / step = ~0.26 ms / chip / step.
- Stacked with steps 1-4 = **~12.6 ms aggregate / step saved =
  ~0.39 ms / chip / step.**

### 6.2 Trace-vs-eager analysis

Per V2-tracy-2 finding §1289-1300, **per-op device-time savings do
NOT translate 1:1 to real-loop traced wall-clock**. V2-14
delivered −25.2 ms / step at the 32-chip aggregate (−0.79 ms /
chip / step) but only −0.64 ms / step at real-loop trace
(62.74 → 62.10). The dispatch overhead absorbs most of the
device-time win.

If V2-grid lands the same 4:1 device-vs-wall-clock attenuation:
- Steps 1-4 device delta = 0.13 ms / chip / step → traced delta ≈
  0.13 / 4 ≈ **0.03 ms / step** → real-loop saving ~0.5 % →
  62.10 → 61.6 ms / step → 16.10 → 16.23 tok/s/user. **Not
  meaningful.**
- Steps 1-5 device delta = 0.39 ms / chip / step → traced delta
  ≈ 0.10 ms / step → real-loop saving ~1.5 % → 62.10 →
  61.2 ms / step → 16.10 → 16.34 tok/s/user. **Modest.**

This is a much **smaller** result than the prompt suggests (“10-15
ms / step on 62.10 ms baseline → 80+ tok/s/user”). The “80+
tok/s/user” target requires the device-time saving to fully expose
at trace (no attenuation), which V2-14 data already disproved.

### 6.3 Where the 10-15 ms / step scenario would apply

If V2-grid is run alongside or as a follow-on to V2-17 / V2-18
levers that attack the **launch-count** side of CCL (PERF.md:1334-
1380), then both the device-time AND the host-dispatch overhead
shrink. In that combined regime the device-time delta from the
grid expansion is no longer absorbed by host overhead, and the
2.17× compute area expansion is closer to a wall-clock win.

**Honest framing for the V2-grid follow-up agent:**
- Standalone V2-grid (no other changes) is a **0.5-1.5 % wall-clock
  win** on the current 62.10 ms baseline.
- Combined with a CCL launch-count optimization (V2-18 or similar)
  the grid expansion could enable an additional 3-5 % real-loop
  win as the device time becomes the critical path.
- 80+ tok/s/user (12.5 ms / step) is **not** reachable from the
  V2-grid alone; it would require simultaneous matmul ring
  widening AND CCL launch fusion AND lm_head sharding rework.

### 6.4 Why this is still worth doing

- The 70 idle cores are dark on a 130-core chip — the BH cost
  envelope expects them to be used.
- It’s a **construction-time** change (`sub_core_grids` and a few
  CRS lists), not a kernel rewrite. Low engineering cost relative
  to V2-17 / V2-18.
- It’s a **necessary precondition** for any future op-kernel work
  that wants to span all 130 cores (e.g. a tile-sharded lm_head
  rewrite).
- It enables the BH-130 compute area to actually be saturated,
  which is the prerequisite for any host-overhead optimization to
  show a wall-clock delta.

---

## 7. Sequencing relative to V2-17c

### 7.1 Scope overlap

V2-17c (in flight) is the recurrent DeltaNet update + readout fused
tt-lang kernel (`tt/kernels/recurrent_delta_rule_v2_kernel.py`,
consumer at `tt/qwen36_delta_attention.py:1404-1700`). It uses a
4-core grid at `(2, 5)-(5, 5)` (line 1465-1467) — **inside the
current 60-core `sub_core_grids`**.

The V2-17c kernel does NOT consume `sub_core_grids` from
`model_config`; it consumes a literal `CoreRangeSet` defined
in-block. So V2-17c is **invariant** to the grid widening in this
doc — it will continue to work bit-identically before and after.

The V2-17c kernel does mention `grid="auto"` in its tt-lang author
script, which on a real device resolves to the live
`mesh_device.compute_with_storage_grid_size()` = `(13, 10)` on BH.
So the **kernel itself** would already pick up the BH-130 dim if
its author lifted it from the hard-coded 4-core grid. That is an
*orthogonal* improvement to the V2-17c kernel and out of scope for
V2-grid.

The rest of DeltaNet — `_output_proj_and_reduce`, `_compute_beta_g`,
the `chunk_gated_delta_rule_ttnn` prefill path — consumes program
configs from `model_config["FF1_3_TG_RING_PROGCFG"]` etc., which
DO read `sub_core_grids` via the 24-core ring. Those benefit from
V2-grid step 5 only (the matmul ring widening), not from V2-17c.

### 7.2 Recommendation

Land V2-17c first (whatever its outcome — its current real-loop
delta is +10.39 ms / step / −16.7 % per V2-18 §1483-1494, but the
authoring is in-flight). Once V2-17c is in main, dispatch V2-grid
as a single multi-step task following the order in §4. The two
changes have no shared state and no sequencing constraint beyond
“don’t double-commit a feature branch”.

If V2-17c is **abandoned**, V2-grid is still independently
worthwhile (it touches different code paths). If V2-17c is
**accepted**, V2-grid runs on top and the V2-tracy-3 sheet should
attribute the deltas separately:
- V2-17c attributes to DeltaNet kernel block (its recurrent +
  readout slice)
- V2-grid attributes to lm_head (`MinimalMatmul`) + SDPA + matmul
  ring (if step 5 lands)

The two attributions are non-overlapping per-op, so V2-tracy-3 can
cleanly isolate each lever’s contribution.

---

## Hand-off summary

To execute V2-grid:

1. Do step 0 (live-mesh dim check) **before** modifying any code.
2. Do steps 1-4 in order, gating each with PCC + coherency. Each
   step is a 1-file or 2-file edit (qwen36_model_config.py + the
   relevant test mocks / demos that hard-code the grid).
3. Tracy V2-tracy-3 after step 4. If the real-loop delta is below
   1 ms / step (the honest §6.2 projection), STOP and reconsider
   step 5.
4. Step 5 (matmul ring widening) is the highest-leverage but
   highest-risk change. Treat it as a separate sub-task with its
   own PCC gate and tracy capture.
5. The 60 + 24-core LM head ring and the 24-core FF / XQKV / WO
   ring are the **load-bearing** ring constants. Touching them
   requires re-deriving the padded-width math at qwen36_model_config
   lines 170-172, 740-742, 760, 762, 769, 776, 783.

All numerical claims in this doc are sourced from existing
artifacts (PERF.md V2-tracy-2 attribution, BRINGUP_LOG.md V2-16
real-loop number, `tt/prefetcher_common.py` BH grid comment). The
only unknown that requires live device access is the **exact
DRAM grid** on the target SKU — §1.3, §5.6 flag this for step 0.
