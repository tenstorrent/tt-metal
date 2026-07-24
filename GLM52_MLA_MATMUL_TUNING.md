# GLM-5.2 MLA matmul tuning — working notes (resume point)

Goal: hand-tune `ttnn.linear` program_config + memory_config for **all 9 matmuls** in GLM-5.2's
sparse-MLA forward under **chunked prefill** (seq_len_local=640), following Iva's tuning process.
Scope = MLA only (NOT MoE). Runs on the **8-device BH loudbox** (`/localdev/ipotkonjak/tt-metal`).

## The 9 matmuls (base 6 + 3 indexer)

GLM-5.2 geom (`reference/glm_5_2_config.py`): hidden=6144, heads=64, q_lora=2048, kv_lora=512,
qk_nope=192, qk_rope=64, v_head=256, index_n_heads=32, index_head_dim=128.
Per-chip after SP/TP=4 (M=640, M_t=20 for all):

| # | matmul | Z(batch) | M | K | N | tiles | in0(act) | in1(wt) | out |
|---|--------|---|---|---|---|-------|----------|---------|-----|
| 0 | q_a_proj | 1 | 640 | 1536 | 2048 | 61k | BF16 | BF8 | BF16 |
| 1 | q_b_proj | 1 | 640 | 2048 | 4096 | 164k | BF16 | BF8 | BF16 |
| 2 | wkv_b1 (batched) | 16 | 640 | 192 | 512 | 31k | BF16 | BF8 | BF16 |
| 3 | kv_a_proj_with_mqa | 1 | 640 | 1536 | 576 | 17k | BF16 | BF8 | BF16 |
| 4 | wkv_b2 (batched) | 16 | 640 | 512 | 256 | 41k | BF16 | BF8 | **BF8** |
| 5 | o_proj | 1 | 640 | 4096 | 6144 | 492k | **BF8** | BF8 | BF16 |
| 6 | indexer.wq_b | 1 | 640 | 2048 | 4096 | 164k | BF16 | BF8* | BF16 |
| 7 | indexer.wk | 1 | 640 | 1536 | 128 | 3.8k | BF16 | BF8* | BF16 |
| 8 | indexer.weights_proj | 1 | 640 | 1536 | 32 | 960 | BF16 | BF8* | BF16 |

**\* indexer weight dtype decision:** `indexer.py:148-170` currently hardcodes `dtype=ttnn.bfloat16`
for wq_b/wk/weights_proj. Iva decided to **switch these to BF8** (bfloat8_b) to match the base MLA
weights → the tuning targets BF8. **The indexer.py load must change `ttnn.bfloat16` → `bfloat8_b`
for the 3 indexer weights** (still TODO, see Wiring below). Once BF8, **indexer.wq_b == q_b_proj**
(same shape + dtypes) → one config serves both.

### Validated against the real Galaxy run
`ops_perf_results_2026_07_23_11_18_12.csv` (repo root; recorded on 8x4 Galaxy, glm52). It has
288 Matmul rows / 32 devices = **exactly 9 matmuls per device**. All 9 per-device shapes + op count
MATCH. Dtypes above are taken from that CSV (except the indexer BF16→BF8 decision).

## Test file

`models/demos/deepseek_v3_d_p/tests/op_unit_tests/test_mla_matmuls_glm_chunked.py`
- Mesh `(2,4)`: sp_axis=0 (size 2) × tp_axis=1 (size 4) = all 8 devices. Global seq 2*640=1280 →
  per-chip 640 (M_t=20) — reproduces one production 8x4 chip. Grid capped 11×10.
- `SHAPES` dict (shape/sharding/tp_out_mode per matmul), `IN0_DTYPE` (o_proj=BF8, rest BF16),
  weights all BF8. `_reconstruct` rebuilds the global tensor from the mesh for PCC (handles
  sum / shard_n / shard_heads / replicated tp reductions).
- `test_glm_mla_mm` runs the `BEST` config per matmul (PCC>=0.99 vs torch). `test_glm_mla_mm_sweep`
  runs `SWEEP` candidate variants for a single tracy pass.
- All 9 PCC ~0.9999.

## How to run + profile (on loudbox)

```bash
source python_env/bin/activate
export TT_METAL_CACHE=/localdev/ipotkonjak/tt-metal-cache   # /home weka mount is 9.4G, 100% FULL
# PCC only:
python3 -m pytest models/demos/deepseek_v3_d_p/tests/op_unit_tests/test_mla_matmuls_glm_chunked.py::test_glm_mla_mm -v
# tracy device perf -> ops_perf_results CSV under generated/profiler/reports/<ts>/:
python3 -m tracy -r -p -m "pytest models/demos/deepseek_v3_d_p/tests/op_unit_tests/test_mla_matmuls_glm_chunked.py::test_glm_mla_mm"
```

Env gotchas:
- **`TT_METAL_CACHE` MUST be redirected** off `/home` (weka mount full) or the profiler build fills
  the disk mid-run ("No space left on device" building instrumented kernels).
- The profiler needs the **correct firmware** (Iva updated it) — with the wrong FW, tracy emits raw
  device timestamps (~4.6e12) instead of durations in `DEVICE KERNEL DURATION [ns]`.

### Parsing the CSV (op order is scrambled + same-shape variants collide)
Map each matmul row via the **`ATTRIBUTES`** column (full `program_config` string: in0_block_w,
per_core_M/N, subblocks, config type) + `INPUT_0_MEMORY`/`INPUT_0_DATATYPE` + `OUTPUT_0_MEMORY`, NOT
by row order. Rows are 8 per op (one/device); average `DEVICE KERNEL DURATION [ns]` over the 8.
util% = tiles*32 / CORE_COUNT / 1.35 / measured_ns * 100.

## Tuning results (per-chip, seq 640, HiFi2, grid 11×10) — the BEST configs

| matmul | program_config | act | out | cores | us | util% |
|--------|----------------|-----|-----|-------|-----|-------|
| o_proj | MultiCast2D ib16 pcm2 pcn18 sub1x6 | DRAM | **L1** | 110 | 135.4 | 78 |
| q_b_proj | MultiCast2D ib8 pcm2 pcn12 sub1x6 | L1 | L1 | 110 | 49.1 | 72 |
| indexer.wq_b | == q_b_proj | L1 | L1 | 110 | 49.1 | 72 |
| wkv_b2 | Reuse ib2 pcm4 pcn8 sub4x2 | L1 | L1 | 80 | 51.7 | 23 |
| wkv_b1 | Reuse ib6 pcm4 pcn16 sub2x4 | L1 | L1 | 80 | 49.8 | 18 |
| q_a_proj | MultiCast2D ib8 pcm2 pcn6 sub1x6 | DRAM | L1 | 110 | 22.7 | 58 |
| indexer.wk | MultiCast2D ib8 pcm2 pcn1 sub1x1 | DRAM | DRAM | 40 | 13.9 | 16 |
| kv_a_proj_with_mqa | MultiCast2D ib8 pcm2 pcn2 sub1x2 | L1 | L1 | 90 | 11.2 | 41 |
| indexer.weights_proj | MultiCast2D ib8 pcm2 pcn1 sub1x1 | DRAM | DRAM | 10 | 10.2 | 22 |
| **TOTAL** | | | | | **~393 us** | |

(`BEST` dict in the test file is the source of truth for exact configs.)

### Key measured findings
- **Real gains: o_proj** (out→L1 + in0_block_w 8→16: ~149→135 us) and **wq_b** (act/out→L1:
  63→49 us, once BF8).
- **wkv_b1 / wkv_b2 are DM-bound, NOT tunable via program_config.** Config-invariant at ~50 us
  across 6 variants (in0_block_w, subblocks, act/out mem, 1D-vs-non-mcast). Core count is capped at
  **80** (B·M_t=320; pcm=4→80, pcm=3 doesn't divide 320, pcm=2→160 > 110-core grid). The BF16 output
  (wkv_b1 writes ~10.5 MB) is the bottleneck — an op-level change, not a config one.
- **indexer.wk / weights_proj** are core-count-limited by tiny N (N_t=4 → 40c, N_t=1 → 10c). Floor.
- Batched matmuls MUST use `MatmulMultiCoreReuseProgramConfig` (non-mcast). The 1D-mcast path
  serializes to 5 cores (258/330 us).

## Wiring status (task 5 — IN PROGRESS)

**DONE:** `mla.py::_resolve_mm_cfg` now accepts a **list of candidate configs** per seq_len slot (not
just a single dict), picking the first whose `num_heads`/`q_lora_rank`/`chunked_only` tags match this
ttMLA. This lets the 640 slot hold BOTH the Kimi (q_lora 1536) and GLM-5.2 (q_lora 2048) configs.
Backward-compatible (dict entries still work).

**TODO:**
1. `mla_config.py MLA_MATMUL_CONFIG`: for the 6 base weights, change the `640:` value from the single
   Kimi dict into a **list** `[<kimi dict>, <glm dict>]`. GLM dict tagged `num_heads=64,
   q_lora_rank=2048, chunked_only=True` with the tuned program_config/act/out/dtype from BEST above.
2. Indexer linears (`indexer.py` forward: wq_b L570, wk L480, weights_proj L584): they currently pass
   NO program_config (DRAM + auto). Wire the tuned configs. Need a config source (e.g. add a small
   `DSA_INDEXER_MM_CONFIG` in mla_config.py keyed by seq_len, or read MLA_MATMUL_CONFIG). wq_b config
   == q_b_proj. **Also switch the 3 indexer weight loads `ttnn.bfloat16` → `bfloat8_b`** (indexer.py
   ~L150-153, L163-166) per the dtype decision — must land WITH the tuned configs.
3. Gating check: ensure Kimi (q_lora 1536) + DeepSeek (128 heads) still resolve their own/no config
   after the list change (the tag match handles it, but run test_mla / test_sparse_mla to confirm).
4. Validate on device: `test_mla.py` chunked GLM path (or test_sparse_mla) — but note per memory,
   full GLM chunked needs 32-chip Galaxy; validate what's possible on the 8-chip loudbox.

## Scratch / artifacts
- Shape+theoretical scratch: (session scratchpad) `glm52_mla_mm_shapes.md`
- Baseline tracy CSV (pre-tune, correct dtypes for 5/9): `reports/2026_07_23_13_55_29/`
- Full sweep CSV: `reports/2026_07_23_14_32_15/` ; round-2 (batched core test): `.../14_43_17/`
- Final BEST CSV: `reports/2026_07_23_14_45_37/`
- Galaxy reference: `ops_perf_results_2026_07_23_11_18_12.csv` (repo root)

Related memories: [[project_mla_matmul_tuning_chunked]] (the Kimi 640 predecessor),
[[project_glm_mla_consideration_model]], [[feedback_minimal_comments]], [[feedback_no_speculation_measure]].
