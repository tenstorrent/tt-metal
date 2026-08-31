# VSA v0 implementation plan

## Status (2026-08-31, end of first session)

| Item | State | Evidence |
|---|---|---|
| Machine bring-up | ✅ | bare-metal build + venv; eth link retrained via glx_reset; smoke test PCC 99.9995% |
| R1 host geometry | ✅ | `vsa_geometry.py`; 14 CPU tests incl. bit-exact vs vendored upstream at 4 shapes |
| Topk spike | ✅ resolved | `topk_large_indices` exact, 0.83 ms at [1,14,226,1808] k=192; no fallback needed |
| R4 `vsa_sdpa` op | ✅ | 11 device tests: m∈{1,3}, ragged/non-uniform/dense/single-block rows, 14 heads |
| R3 coarse stage | ✅ | `vsa_stages_minimax_h3.py`; 4 device tests vs oracle (sp=1), + coarse+fine e2e |
| R2 KV all-gather | ✅ | persistent-buffer AG in the attention VSA branch (4x8 tests exercise it) |
| R5 attention path | ✅ | 4x8: sparsity0≡ring PCC 99.98%; vs torch oracle 99.5% (gate/placement matrix) |
| R5 traced block | ✅ | 15s/768p block traced+untraced, full gate branch, finite output |
| R6b/c production | ✅ | 15s/768p PCC 99.41%, 1280x768/39f 99.31% (zero gate), 99.35% (random gate) |
| R5 model+pipeline plumbing | ✅ code, model tests pending | pack/unpack gathers in model forward; `_prepare_vsa` + tiled metadata in pipeline |
| R6a/R6d model level | tests written, running | `test_vsa_transformer_minimax_h3.py` |
| Full pipeline run w/ VSA | ⬜ untested | plumbing in place (`MiniMaxH3Pipeline(vsa_config=...)`); needs a real-checkpoint generation run |
| R6a "config off bit-identical" | by construction + dense regression test | every change branches on `vsa_config`/geometry being set |

Residual PCC vs the fp32 torch oracle (~99.3-99.5% on random data) is dominated by near-tie
top-k selection across the bf16/fp32 boundary — index sets are compared as sets with ties
tolerated per the scope; the dense-set comparison (sparsity 0) sits at 99.98%.

Companion to `VSA_SCOPE.md` (requirements). Facts below were surveyed from this tree; upstream
FastVideo reference files are vendored in `vsa_reference/` (from
github.com/hao-ai-lab/FastVideo @ main, 2026-08-31).

## Key surveyed facts

### Attention (`attention_minimax_h3.py`)
- Three existing paths chosen in `forward` (no config object): `use_exp_ring_sdpa` (L192),
  `use_ring` (L130), local dense fallback. Tests poke attributes directly.
- `forward(spatial_1BND, N, rope_cos, rope_sin, addcmul_residual, addcmul_gate)`;
  QK-norm + head-split + RoPE fused in `self.norm_q/norm_k(...)` → `[1, n_local_heads, N_local, 128]`;
  V via `nlp_create_qkv_heads`. After attention: `concatenate_heads` → to_out (ColParallel, TP
  gather ahead of/fused into matmul).
- `to_qkv`/`to_out` are `ColParallelLinear` (`models/tt_dit/layers/linear.py`). Zero-init of a
  missing checkpoint key belongs in `MiniMaxH3Attention._prepare_torch_state` via
  `state.setdefault("to_gate_compress.weight", torch.zeros(inner_dim, hidden_size))`
  (torch [out,in] convention; precedent `transformer_motif.py:436`).
- `agmm_config.py` `AGMM_BLOCK_SIZES` lacks (5376, 1792); K_block must divide K/32/tp = 42 →
  add an entry for the gate matmul.
- Compute configs on attention: `sdpa_compute_kernel_config` (HiFi2), `mm_compute_kernel_config`
  (HiFi2 + fp32 acc) — use the latter for pooling matmuls. `sdpa_worker_grid` excludes last
  column (CCL).

### R2 all-gather
- No standalone K/V AG today (ring ops gather internally). Use
  `ccl_manager.all_gather_persistent_buffer(k_BHNE, dim=2, mesh_axis=sp_axis)`
  (`models/tt_dit/parallel/manager.py:823`); ping-pong buffer cache key ("ag", shape, dim, axis,
  dtype) — distinct from ring's key, no collision.

### Host packing (`models/tt_dit/pipelines/minimax_h3/packing.py`)
- `build_packed_sequence` (L243) lays out `[text | condition | audio | video]`;
  tags VIDEO=0/TEXT=1/AUDIO=2; `build_rope_tables` L328; `adaln_indices` L378.
- Upload points in `pipeline_minimax_h3.py`: `_device_metadata` L983 (rope), `_row_indices` L1009
  (adaln), fracture at `mesh_partition(hidden, 2, sp_axis)` in transformer forward L496;
  `padded_len = ceil(seq_len / (sp*TILE)) * sp*TILE` L1765.
- New VSA config must be explicit kwargs through `_prepare_transformer` (L835) →
  `MiniMaxH3Transformer3DModel` → block → attention (config.json splat would TypeError).

### Upstream oracle (vendored in `vsa_reference/`)
- `_h3_tile_geometry(prefix_segments, dit_seq_shape, device, tile_shape=(4,4,4))` →
  (tile_partition_indices, variable_block_sizes, untile_combined_index, num_prefix_tiles,
  num_video_tiles). Video tiles cube-major via `get_tile_partition_indices`; valid counts via
  `construct_variable_block_sizes`; packed row → padded slot map `untile_combined_index`.
- Selection (exempt mode, ours): k_vid = max(1, min(ceil((1−s)·V), V)) over video cols only;
  prefix cols always selected; prefix-query rows fully dense. `_build_block_mask` L386.
- Pooling `_pool_tiles`: masked mean = sum(64 slots, pads zero) / valid_count, fp32.
- scores = q_pool @ k_poolᵀ / √d. Gate: out = out_fine + gate ⊙ (softmax(scores) @ v_pool
  broadcast tile→rows). Token-level oracle `reference_sparse_attention` in
  `vsa_reference/test_vsa_h3_metadata.py:46`.

### sparse_sdpa_msa (fork base for R4), all under `ttnn/cpp/ttnn/operations/transformer/sdpa/`
- Fork: `sparse_sdpa_msa.{hpp,cpp}`, `device/sparse_sdpa_msa_device_operation{_types.hpp,.hpp,.cpp}`,
  `device/sparse_sdpa_msa_program_factory.cpp`, `device/kernels/dataflow/sparse_sdpa_msa_{reader,writer}.cpp`,
  `device/kernels/dataflow/sparse_sdpa_msa_gather.hpp`, `device/kernels/compute/sparse_sdpa_msa_compute.cpp`.
  Registration: `sdpa_nanobind.cpp:407` + `sources.cmake:24-26,49` + `CMakeLists.txt:35`.
- Shared helpers to reuse (not fork): `dataflow_common.hpp` (fill_neginf_tile,
  fill_vertical_tile_bf16), `compute_common.hpp` (apply_padded_mask_lightweight_runtime,
  apply_partial_mask_lightweight), `compute_streaming.hpp` (blocked_matmul_and_pack,
  reduce_c_row_group, sub_exp_block_bcast_cols, sub_exp_first_col_blocks, salad_correct_fused,
  normalize_row_streaming), tilize/untilize helpers.
- MSA semantics vs ours: MSA work unit = one query *token* across GQA heads (q ROW_MAJOR,
  heads-as-rows); indices `[1, n_kv, S, TOPK]` per token; k_chunk == block_size; sentinel tail
  found by binary search; invalid columns never fetched (no numeric masking needed except causal).
  Dual-NoC split gather via cb_kreq/cb_kack + TridRing. Work = S*n_kv split contiguously over
  full grid; dispatch args patched by `override_runtime_arguments` (arg enums in device_op hpp).
- vsa_sdpa deltas: q TILE `[1,H,S_local,d]` head-major; work unit = (head, q-tile of 64 tokens =
  2 q tile-rows); indices `[1, H, S_local/64, T/64]`; block_size=64 (Skt=2); m blocks per L1
  chunk (partial last chunk); `block_counts [T/64]` uint32 → per-block column masking
  (cols ≥ count → −inf) applied within chunks (ragged tiles DO need numeric masking, unlike MSA);
  non-causal only; no block-cyclic, no cache_batch_idx, no chunk_start_idx.

### ttnn.topk (R3 spike)
- Composite `ttnn.topk` on Blackhole bf16 routes k∈(64,2048] to
  `ttnn.experimental.topk_large_indices` (k rounded to mult of 16; ROW_MAJOR bf16 in; uint32
  indices out; optional valid_length; width ≤ 2^30). k≈200 over 1802 cols → pad cols, k=208 or
  ask 200→uses 208? route requires k%16==0 after rounding; slice back. Model precedent:
  `models/demos/minimax_m3/tt/attention/msa.py:147` uses topk_large_indices directly.

## Production geometry (for reference)
- 15s/768p: TBD exact latent grid; test fixture `_PROD` upstream: latents (37,48,84), patch
  (1,2,2) → token grid (37,24,42) = 37296 video tokens, prefix 300 text + 414 audio.
  Scope says ~1802 total tiles, k≈200 at sparsity 0.9.
- Mesh: TP=4 (axis 0), SP=8 (axis 1); 14 local heads × 128; N multiple of 8·64.

## Work plan (order)
0. Smoke test dense block on 4×8 (build + venv + `test_minimax_h3_transformer_block`).
1. **R1 host geometry** `vsa_geometry.py` (new, models/tt_dit/pipelines/minimax_h3/):
   port upstream builders + SP-aware packing: pad packed sequence to tiles ON HOST in tile order
   (scope: per-device seq multiple of 64, no tile straddles shard), tags, placement maps
   (identity, striped), permuted rope/adaln/unpack. Torch-oracle unit tests (CPU only) mirroring
   `vsa_reference/test_vsa_h3_metadata.py`.
2. **Topk spike** (device): ttnn.topk / topk_large_indices at [1,14,~1802 rows?,1802 cols] — per
   (head, q-tile) rows. Decide topk vs threshold bisection; record in this doc.
3. **R3 coarse stage** as ttnn composition + unit test vs oracle (pool matmul, AG, scores,
   softmax@Vc, gate linear, topk + index assembly to R4 layout with sentinels).
4. **R4 vsa_sdpa** fork: scaffold (types/validate/factory), reader (indexed m-block gather +
   count masking), compute (online softmax over chunks), writer; python tests vs torch reference
   (m=1 and m>1, ragged, non-uniform rows, fully-dense rows, single-block rows).
5. **R2 + R5 integration**: KV AG, fourth path in attention behind explicit kwargs
   (vsa_sparsity, vsa_tile_placement, vsa_k_chunk_multiplier), plumb through block/model/pipeline;
   traced + untraced.
6. **R6 gates**: sparsity=0 vs ring path; 0.9 vs oracle (both shapes); nonzero gate; striped ≡
   identity.

## Decisions log
- **2026-08-31 machine setup**: 4x8 BH galaxy runs bare-metal (no docker): build_metal.sh with
  clang-20, create_venv.sh, pinned diffusers (abc5e9bf71) for the torch reference. Requires
  `TT_MESH_GRAPH_DESC_PATH=tt_metal/fabric/mesh_graph_descriptors/single_bh_galaxy_torus_xy_graph_descriptor.textproto`
  (see run_h3_test.sh). The machine had one untrained eth link (chips 1<->25, torus wrap) which
  broke FABRIC_1D_RING topology mapping; `tt-smi -glx_reset` retrained it. Smoke test
  `test_minimax_h3_transformer_block[blackhole-small_s2048-4x8...]` passes at PCC 99.9995%.
- **R3 topk spike (resolved)**: at production selection shape [1, 14, 226 rows, 1808 cols] bf16,
  k=192 (179 rounded to k%16==0): `ttnn.experimental.topk_large_indices` is correct (index sets
  match torch exactly on sampled rows) and takes 0.83 ms/iter, uint32 out; composite `ttnn.topk`
  also correct at 1.63 ms/iter but returns uint16 indices. Decision: use topk_large_indices, slice
  to exact k on host side of the index assembly; no threshold-bisection fallback needed.
  (models/tt_dit/tests/models/minimax_h3/vsa_topk_spike.py)
- **R4 kernel structure**: work unit = flat index w over H*(S/64) rows; head-major layout makes
  q/out/index page addressing `w * tiles_per_work`. Fixed-size CB batches (m blocks per chunk,
  m mask slots) keep reader/writer L1 offsets stable; partial last chunks leave tail tiles
  unread (runtime chunk width in compute; no zero-fill, no masking of absent blocks). Ragged
  blocks masked via count-derived partial-column tiles (slot b = block b) + neginf stamps,
  L1-accumulated onto scores before the row-max reduce.
- **Pad rows carry finite garbage**: the tiled sequence flows through the whole model, so pad
  slots are nonzero by the time they reach attention; correctness relies on count masking
  (K side), the averaging matrix (pooling), and unpack dropping pad rows (Q side) — never on
  pad values. -inf mask via L1-acc add requires finite (non-NaN) inputs.
