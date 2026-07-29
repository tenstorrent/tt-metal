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
| 6 | indexer.wq_b | 1 | 640 | 2048 | 4096 | 164k | BF16 | BF8 | BF16 |
| 7 | indexer.wk | 1 | 640 | 1536 | 128 | 3.8k | BF16 | BF8 | BF16 |
| 8 | indexer.weights_proj | 1 | 640 | 1536 | 32 | 960 | BF16 | **BF16** | BF16 |

**Indexer weight dtype — resolved by #51005 (merged to main):** wq_b and wk switched
`ttnn.bfloat16` → `bfloat8_b` in `indexer.py`, as anticipated here. `weights_proj` did **NOT** —
it stays BF16: the per-head gate it feeds is precision-sensitive, unlike wq_b (drives top-k
selection only, tolerates rounding) and wk (immediately LayerNorm'd, which cancels the BF8
magnitude error). Since wq_b is now BF8, **indexer.wq_b == q_b_proj** (same shape + dtypes) → one
config serves both. `weights_proj`'s BF16 weight means it does NOT share q_b_proj's tuning and
needs its own re-check (see Tuning results below — its BEST config was previously validated only
at BF8 weight).

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
| indexer.weights_proj | MultiCast2D ib24 pcm2 pcn1 sub1x1 | **L1** | **L1** | 10 | 6.2 | 37 |
| **TOTAL** | | | | | **~389 us** | |

(`BEST` dict in the test file is the source of truth for exact configs.)

### Key measured findings
- **Real gains: o_proj** (out→L1 + in0_block_w 8→16: ~149→135 us) and **wq_b** (act/out→L1:
  63→49 us, once BF8).
- **wkv_b1 / wkv_b2 are DM-bound, NOT tunable via program_config.** Config-invariant at ~50 us
  across 6 variants (in0_block_w, subblocks, act/out mem, 1D-vs-non-mcast). Core count is capped at
  **80** (B·M_t=320; pcm=4→80, pcm=3 doesn't divide 320, pcm=2→160 > 110-core grid). The BF16 output
  (wkv_b1 writes ~10.5 MB) is the bottleneck — an op-level change, not a config one.
- **indexer.wk / weights_proj are core-count-limited by tiny N** (N_t=4 → 40c, N_t=1 → 10c — a hard
  geometry floor, `per_core_M` can't go below `ceil(M_t/grid.y) = ceil(20/10) = 2` either, so 10
  cores is the max reachable for weights_proj on this grid). **weights_proj re-tuned (2026-07-29)**
  for its #51005 BF16 weight: the old BEST (DRAM/DRAM, ib8, 10.2 us/22%) was still tuned for the
  BF8-weight era. Swept act/out mem + in0_block_w (8 variants): moving **activation** to L1 was the
  real lever (act is 640×1536 BF16 ≈ 1.97 MB vs the weight's mere 98 KB — out-mem placement barely
  moved it, DRAM act alone stayed ~10 us regardless of out mem). Best: L1/L1 + `in0_block_w=24` →
  **6.2 us, 37%** util (ib bracket 8/12/16/24/48 → 7.62/6.73/6.29/6.17/6.66 us, peak near 24). Still
  well under 100% util since cores are floored at 10 — not fully tunable away, just improved.
- Batched matmuls MUST use `MatmulMultiCoreReuseProgramConfig` (non-mcast). The 1D-mcast path
  serializes to 5 cores (258/330 us).

## Wiring status (task 5 — DONE, program_configs; act/out L1 placement is next)

**Config plumbing:** `mla_config.py MLA_MATMUL_CONFIG`'s `640:` slot for the 6 base weights is now a
**list** `[<kimi dict>, <glm dict>]`; the GLM dict is tagged `num_heads=64, q_lora_rank=2048,
chunked_only=True` with the tuned program_config/act/out/dtype from BEST above. Added 3 new
top-level keys (`indexer.wq_b`, `indexer.wk`, `indexer.weights_proj`) for the DSA indexer linears —
tagged `num_heads=64, q_lora_rank=2048` (no `chunked_only`: the indexer's write_k/forward are always
block-cyclic, single-shot folds onto the same shape, so there's no separate single-shot shape to
exclude). The candidate-list resolution logic is now a shared function,
`mla_config.resolve_gated_matmul_config`, used by both `ttMLA._resolve_mm_cfg` (refactored to a thin
wrapper) and a new `TtIndexer._resolve_mm_cfg` in `indexer.py`. `indexer.py`'s three `ttnn.linear`
calls (write_k's `wk`, forward's `wq_b`/`weights_proj`) now pass `program_config` + `memory_config`
from the resolved config when one applies (defaulting to the old DRAM/auto behavior otherwise).
**Note:** `wq_b`'s output keeps its existing hardcoded `dtype=bfloat8_b` regardless of the resolved
config's `out_dtype` (BF16) — that's a deliberate downstream-scoring requirement unrelated to the
matmul-tuning test's measured shape, so the config's `out_dtype` is intentionally NOT applied there.

**Bug caught by an isolated gating sanity check before touching hardware:** the indexer entries were
first written with `**_GLM_TAGS` (which includes `chunked_only=True`), silently excluding them
whenever a caller wasn't built `is_chunked=True` — contradicting the "no chunked_only" comment.
Fixed with a separate `_GLM_INDEXER_TAGS` (no `chunked_only`). Caught by calling
`resolve_gated_matmul_config` directly in a throwaway script and asserting the expected candidate
came back for each (weight, num_heads, q_lora_rank, is_chunked) combination — before any device run.

**act/out L1 placement — NOT done, and needs a decision, not a mechanical wire-up:** program_config +
out_mem_config are fully local per matmul (safe to set unconditionally). ACT (input) residency is
different: `hidden_states` (post-attn_norm) is a **shared** input to 4 of these matmuls —
q_a_proj (DRAM-tuned) and indexer.wk (DRAM-tuned) want DRAM; kv_a_proj_with_mqa and
indexer.weights_proj (both L1-tuned) want L1 — and its residency is set by the CALLER
(`TtPrefillBlock`'s attn_norm output config), not by mla.py/indexer.py (see the existing "hidden
states memory config is set outside the module" comments at `mla.py` `_q_a_latent`/`_kv_stem`).
Forcing it either way helps two consumers and hurts the other two; q_a_proj is by far the biggest of
the four (22.8us vs 11.2/13.8/6.2us) so DRAM (today's default) is likely still the right overall
choice, but that's a measured trade-off to make, not a default to flip. The **qr** latent (feeding
q_b_proj + indexer.wq_b) has NO such conflict — both want L1 and qr's residency is already
auto-wired via the existing `_get_act_mem_config("q_b_proj", ...)` -> `norm_memory_config` mechanism,
so it needs no further work once the q_b_proj GLM config entry exists (it now does).

**Validated on the 2x4 Blackhole loudbox** (new test `test_sparse_mla_chunked_mm_tuned_shape` in
test_sparse_mla.py, seq_len=1280/chunk=1280 -> per-chip seq_len_local=640, the exact tuned shape —
nothing else in this file hits it, SPARSE_ANCHOR_CASES runs seq=5120/chunk=1024 -> local=512):
glm_5_1 + glm_5_2 (now resolving the wired GLM configs) and deepseek_v32 (correctly falls through to
`None` / defaults — num_heads=128/q_lora=1536 matches neither Kimi nor GLM tags) all pass against the
CPU reference. Full `test_sparse_mla.py` regression: 35 passed, 0 failed. Full Galaxy 8x4 validation
is out of reach here (per memory, needs the 32-chip machine).

**Gotcha hit along the way:** switching branches (from the KV-TP-sharding branch, rebased onto much
newer `main`, to this one, based on older `d4e579756ad`) without rebuilding caused the FIRST test
that actually exercised `update_padded_kv_cache`'s reader kernel (`test_sparse_mla_chunked_mm_tuned_shape`
new test) to fail with a kernel-compile error (`reader_update_padded_kv_cache.cpp`, a
`TensorAccessorArgs` mismatch) — all 3 variants failed identically including deepseek_v32, which my
config changes don't even touch, so it was clearly a build/branch mismatch, not a real bug. Fixed by
`./build_metal.sh` on this branch before retrying. Matmul-only tests (`test_glm_mla_mm`) never
touched this kernel so didn't surface it earlier.

**Still open:** the act/out L1 placement decision above; production adapter/runtime-config wiring
(same scope note as the KV-TP-sharding workstream — out of scope here too); Galaxy 8x4 validation.

## Scratch / artifacts
- Shape+theoretical scratch: (session scratchpad) `glm52_mla_mm_shapes.md`
- Baseline tracy CSV (pre-tune, correct dtypes for 5/9): `reports/2026_07_23_13_55_29/`
- Full sweep CSV: `reports/2026_07_23_14_32_15/` ; round-2 (batched core test): `.../14_43_17/`
- Final BEST CSV: `reports/2026_07_23_14_45_37/`
- Galaxy reference: `ops_perf_results_2026_07_23_11_18_12.csv` (repo root)

Related memories: [[project_mla_matmul_tuning_chunked]] (the Kimi 640 predecessor),
[[project_glm_mla_consideration_model]], [[feedback_minimal_comments]], [[feedback_no_speculation_measure]].
