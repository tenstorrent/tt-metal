# GLM-5.3-Flash Phase 2: multi-chip decoder (work in progress, 2026-10-20)

Design of record: `~/ashai/GLM_5.3_DESIGN.md` sections 4-5 (columns = TP group of a row, rows = DP
user groups and the expert-TP axis). Code: `tt/mesh.py` (`GLMMesh`, placement helpers,
collectives) threaded through every stage-1 module with a `mesh=` keyword; a `(1, 1)` mesh keeps
the single-chip behaviour (every collective is the identity, every placement replicated).

## Layout per module (as implemented)

| module | placement on a `(rows, cols)` mesh | collective |
|---|---|---|
| mHC, norms, router, indexer | replicated | none |
| dense MLP / shared expert | `[gate_c \| up_c]` of w13 per column, rows `c` of w2 | `all_reduce` over columns (shared expert folded into the routed partial first) |
| routed experts | experts `[c*E/cols, (c+1)*E/cols)` per column, intermediate slice `r` per row; per-column expert selector `[E, E/cols]` (identity block) picks the local routing weights | `all_gather` tokens over rows -> local `sparse_matmul` -> `reduce_scatter` over rows -> `all_reduce` over columns |
| KDA decode | heads per column (head-major `[q_c\|k_c\|v_c]` projection, taps, gates, A_log/dt_bias); state `[B, 64/cols, 128, 128]` + conv `[B, 3, 3*2048]` per chip | `all_reduce` over columns after o_proj |
| KDA prefill | ttKDA with `tensor_parallel_axis=1` + `TT_CCL` handles (reduce-scatter on hidden, then `all_gather` over columns); single-row meshes only (multi-row = sequence-parallel ttKDA, the Phase-3 prefill engine) | ttKDA's own reduce-scatter + `all_gather` |
| DSA | q_a/kv_a/norms replicated; q_b, kv_b (absorbed and prefill forms), o_proj head-sharded over columns (16 heads per chip at TP4); latent cache + pooled-key index replicated across the columns (Phase-2 first step; user-partitioned cache next) | `all_reduce` over columns after o_proj |
| positions / streams | `[rows*B]` positions and `[1,1,rows*B,16384]` streams row-sharded (row r = users `[rB, (r+1)B)`) | none |

Not wired yet (the decoder refuses them): prefill on multi-row meshes; sparse DSA decode (positions
> 2050) on multi-row meshes (needs the metadata-tensor variants so every row runs one SPMD graph);
the `sparse_sdpa` kernel wants H % 32 == 0 (TT_FATAL at 16 local heads): the query is padded with zero heads to 32 and the padded outputs dropped (`TtDSA._pad_heads_rm`), i.e. 2x attention compute on the sparse rows at TP4 until the user-partitioned layout (all 64 heads on the cache owner) lands.

## Test beds on the 32-chip Galaxy (disjoint from the single-chip work on chips 0-8)

| mesh | env | probe |
|---|---|---|
| (2, 4) block, grid rows 4-5 | `TT_VISIBLE_DEVICES=27,26,31,30,23,22,19,18` + `p150_x8_mesh_graph_descriptor.textproto` + `MESH_DEVICE=P150x8`, FABRIC_2D RELAXED_INIT, max payload router config, `l1_small_size 32768` | `~/g4cov/mesh_sanity.py 2 4` PASS (`_logs/mesh/sanity_rows45.log`) |
| (1, 4) ring, grid row 6 | `TT_VISIBLE_DEVICES=25,29,21,17` + `~/g4cov/p150_x4_galaxy_2ch.textproto` + `MESH_DEVICE=P150x4` | `mesh_sanity.py 1 4` PASS (`sanity_row6_1x4.log`) |

`TT_METAL_WATCHER` cannot be combined with the fabric kernels on this build (mesh open fails with
`program.cpp:3760 state.offset <= max_size`); `pytest.ini` sets a 300 s timeout, the mesh tests carry
`@pytest.mark.timeout(1800)` (weights + first compile of a MoE layer exceed 300 s).

## Evidence (real weights, PCC vs HF; logs under `~/ashai/glm53/_logs/mesh/`)

| test | mesh | result | log |
|---|---|---|---|
| `test_mesh_ccl.py` | (2,4) | TP SwiGLU + `all_reduce_cols` vs torch per row-chip > 0.995 (3 iterations, rows carry different tokens, columns byte-equal); `all_gather_rows` bf16/fp32 byte-exact; `reduce_scatter_rows` 0.9999; the three collectives inside a captured trace, 3 replays on new inputs > 0.9999 | `test_mesh_ccl_1.log` |
| `test_mesh_moe.py` (layer 3) | (2,4) | hybrid MoE vs HF MoE: T=2/row 0.999959, T=32/row 0.999967, replicated | `test_mesh_moe_1.log` |
| `test_mesh_decoder.py` layer 0 | (2,4), B=2/row | 4 users x 48 prompt tokens through the captured decode trace: per-user 0.99991-0.99993 (worst single row 0.99984); HF-cache traced steps 48-50: 0.99989-0.99994 | `test_mesh_decoder_l0_1.log` |
| layer 3 | (2,4), B=2/row | per-user 0.99974-0.99989 (worst single row 0.99652 = the position-0 row); traced steps 48-50: 0.99993-0.99997 | `test_mesh_decoder_l3_2.log` |
| layer 4 | (2,4), B=2/row | per-user 0.99994-0.99999 (worst row 0.99789); traced steps: 0.99996-0.99999 | `test_mesh_decoder_l4_1.log` |
| `test_mesh_tp_prefill.py` layer 0 | (1,4) | TP4 prefill S=200 vs HF 0.999853 (replicated across the 4 chips); decode 200-202: 0.99990-0.99992 | `test_tp4_prefill_l0_3.log` |
| layers 3, 4 | (1,4) | layer 3: prefill 0.999878, decode 200-202: 0.99994-0.99997; layer 4: prefill 0.999978, decode 0.999995-0.999996 | `test_tp4_prefill_l34_1.log` |
| `test_mesh_tp_prefill.py::test_tp4_sparse_prefill_layer3` | (1,4) | layer 3 R4097 single-shot prefill at TP4 after the `seq_shard_axes` fix: all 0.999935, rows < 2048 0.999878, rows >= 2048 0.999859 (before: 0.998949 / 0.999878 / 0.997769); attention sublayer vs HF hook: rows >= 2048 0.999749 (single chip 0.999755) | `test_tp4_sparse_fixed.log`, before: `test_tp4_sparse_l3_2.log`, `test_tp4_attn_level.log` |
| `test_mesh_decoder.py` B=32/row (`GLM53_MESH_B=32 GLM53_MESH_S=12`) | (2,4), 64 users | layer 3: per-user min 0.997864 / p10 0.999176 / median 0.999766 over 12 traced steps, HF-cache traced steps min 0.99650-0.99746 median 0.99993; layer 4: per-user min 0.999909 p10 0.999973 median 0.999991 | `test_mesh_decoder_b32_1.log` (layer 4), `test_mesh_decoder_b32_l3_2.log` |

Method note: the mesh has no prefill path on multi-row meshes yet, so the decode test feeds each
user's prompt through the decode step token by token; HF's prefill output row t is the exact
reference of the decode output at step t (same recurrence / causal attention), then HF decode
steps with a live cache follow. Every step after the warm-up runs through the captured trace (one
eager step in the middle), and the outputs are checked byte-equal across the four columns of a row.

## Full 8x4 Galaxy (native mesh, FABRIC_2D_TORUS_XY; `GLM53_MESH_SHAPE=8,4`)

| test | result | log |
|---|---|---|
| `test_mesh_decoder.py` layer 0, B=2/row (16 users, DP8 x TP4) | per-user pcc vs HF prefill min 0.999869 p10 0.999873 median 0.999913; traced HF-cache steps min 0.99985-0.99988 | `test_8x4_decoder_l0_torus.log` (47 s) |
| layers 3, 4 (hybrid experts EP4 x TP8: 72 experts x 256-wide slices per chip) | layer 3: per-user min 0.999525 p10 0.999659 median 0.999773; layer 4: min 0.999912 p10 0.999966 median 0.999986 (16 users x 48 traced steps + 3 HF-cache steps) | `test_8x4_decoder_l34_torus.log` (132 s incl. weight load) |

## Decode performance on the (2,4) block (2026-10-20, `tracy/README.md`)

`tests/test_perf_mesh.py` (traced decode, Tracy signposts, seed-only DSA cache, labelled eager pass for the
per-module attribution) + `tests/perf_mesh_report.py` (per-device `tt-perf-report` tables from the one ops CSV
of the mesh profiler). Device time of chip (0,0), position 1024, expert weights bfp8:

| layer | users/row (total) | device ms/step | single-layer tok/s per user / aggregate | top shares |
|---|---|---|---|---|
| 4 KDA+MoE | 4 (8) | 7.985 | 125 / 1002 | router composite 42 %, mHC 24 %, experts 18 %, KDA 12 %, CCL 1.7 % |
| 4 KDA+MoE | 32 (64) | 12.448 | 80 / 5141 | router 27 %, KDA 26 %, experts 26 %, mHC 18 %, CCL 1.9 % |
| 3 DSA+MoE | 4 (8) | 7.785 | 128 / 1028 | router 44 %, mHC 24 %, experts 17 %, DSA 9.5 %, CCL 3.0 % |
| 3 DSA+MoE | 32 (64) | 9.940 | 101 / 6439 | router 34 %, experts 30 %, mHC 22 %, DSA 9.1 %, CCL 3.3 % |

Single chip: layer 4 10.76 ms, layer 3 10.87 ms (B=1). The first targets are the replicated exact-fp32 router (8 single-core
`cumsum` = 2.64 ms per MoE layer on every chip), the mHC composite (1.9 ms, 220 ops) and the KDA decode
recurrence at B=32 (3.2 ms); the collectives are < 2 % of the step (all-reduce = ReduceScatter + AllGather,
41-65 us).

## Resident multi-layer model, traced end-to-end decode step (`tt/mesh_model.py`, `tests/test_mesh_model.py`)

Embedding (replicated) -> N `FunctionalDecoder` layers on the mesh -> head (replicated, greedy argmax), the
whole step in one captured trace; the prompt goes through the decode step token by token (HF prefill rows
of the dump chain `L45_S128` are the reference of the output after the last built layer).

| mesh | layers | users/row | chain vs HF (per-user pcc, 32 tokens) | greedy id vs host head | traced step | per layer | log |
|---|---|---|---|---|---|---|---|
| (1,4) row-6 ring, TP4 | 0-5 (3 dense + 3 MoE) | 2 | 0.999253 (through 6 layers) | 100 % | 32.55 ms -> 30.7 tok/s per user, 61 aggregate | 5.43 ms (single chip: 10.8 ms per MoE layer) | `test_mesh_model_ring_l0-5.log` |
| (2,4) rows-6-7 block, bfp4 experts (gate/up AND down; the only way 45 layers fit 8 chips), cumsum router | 0-44 | 4 (8 users) | 0.829 (16 tokens, through 45 layers; see the chain note) | 100 % | 318.3 ms -> 3.1 tok/s per user, 25 aggregate | 7.07 ms | `test_mesh_model_rows67_45L_bfp4_2.log` (45 min incl. 40 min of weight loading) |
| 8x4, all 45 layers, bfp8 | 0-44 | 4 and 32 | pending (needs the whole Galaxy free + the board reset) | | | | |

**TTFT, full model** (same run, `_ttft`): 128-token prompt through the sequence-parallel prefill of all 45 layers +
head: warm 1301 ms, cold 2660 ms (first-call compiles); first greedy token 102647 == host head on the HF rows.

**Chain note:** the chain PCC vs HF after many layers drops (6 layers 0.9993, 24 layers bfp8 0.876, 45 layers bfp4
0.80-0.83) while the greedy token stays identical; whether this is precision accumulation over the 42 MoE layers
(near-tie expert flips compound) or a chain-level defect is being separated with the single-chip streaming full
model (`arg/glm53_fullmodel`, true prefill path, per-layer chain PCC).

Build cost: dense layer 2.7 s, MoE layer 43-48 s on the (1,4) mesh (host fp8 -> bf16 dequant 3-7 s, the rest
is the bfp8 conversion + upload), i.e. ~32 min for 45 layers.

## TP4 prefill timing per layer (`tests/test_mesh_prefill_perf.py`, (1,4) row-2 ring, S=1024, warmed wall)

| layer | kind | TP4 prefill | single chip (README stage 1) | TP4 eager decode at 1024 |
|---|---|---|---|---|
| 0 | KDA+dense | 16.9 ms (60.7 K tok/s) | - | 17.0 ms/step |
| 3 | DSA+MoE | 169.4 ms (6.0 K tok/s) | 325.7 ms | 19.8 ms/step |
| 4 | KDA+MoE | 168.3 ms (6.1 K tok/s) | 297.8 ms | 18.6 ms/step |

A single-row TTFT for a 1K prompt through all 45 layers would be ~7.1 s (3 x 17 + 42 x 169 ms), i.e. the MoE
prefill dominates; the design's prefill engine (SP8 x TP4, dispatch-based experts) is what brings it down,
and it is not wired yet (ttKDA is sequence-parallel on multi-row meshes). Eager decode steps carry host
overhead; the traced step is 5.4 ms/layer (resident-model row above). Log `test_tp4_prefill_perf_row2.log`.

## Sequence-parallel prefill v1 (TTFT path)

`tests/test_mesh_sp_kda.py` on the rows-2-3 (2,4) block: ttKDA with the prompt split over the rows (row r =
slab r of the chunk, `active_seq_len` global, grouped-scan `summary_group_chunks` sized to divide the local
chunk count): output 0.999926 vs HF (per row 0.999923 / 0.999916), final recurrent state 0.999897 and
replicated over the rows, conv history 0.999992 (`test_sp_kda_lb2_2.log`). v1 wiring through
`FunctionalDecoder.prefill_forward` on multi-row meshes: KDA layers sequence-parallel, the hybrid MoE
gathers the slabs as before, DSA layers gather the full chunk over the rows, attend on every row and keep
their slab through a reduce-scatter of identical copies (exact, x 1/rows). Limitation: the KDA carries and
the DSA cache/index of the prefilled user land in slot ``user`` of EVERY row (per-row ownership masks /
scratch page tables are the next step). `tt/mesh_model.py::prefill` + `test_mesh_model.py::test_mesh_model_ttft`
measure the prompt -> first token time.

| `test_mesh_sp_prefill.py` (rows-2-3 block, S=1000, FunctionalDecoder API) | prefill vs HF rows | per row-slab | decode at 1000/1001 vs HF (both rows) |
|---|---|---|---|
| layer 0 KDA+dense | 0.999842 | 0.999830 / 0.999834 | 0.99995 |
| layer 3 DSA+MoE | 0.999981 | 0.999952 / 0.999956 | 0.99995 |
| layer 4 KDA+MoE | 1.000000 | 0.999998 / 1.000000 | 0.99999 |

Log `test_sp_prefill_lb2.log` (first-call walls include compile: 15.1 / 10.8 / 0.7 s).

TTFT, resident model layers 0-5 on the rows-2-3 (2,4) block, prompt 128 tokens (`test_mesh_model_ttft_lb2_l0-5.log`):
warm 300 ms (cold 38.7 s = first-call compiles), chain 0.999479 vs HF rows, first greedy token == host head (1526).

## Optimization 1: router tie-break (2026-10-20)

The exact fp32 top-8 router's first-match tie-break used `ttnn.cumsum` (single-core, 330 us x 8 rounds =
2.64 ms per MoE layer, replicated on every chip). Replaced by a second max over the candidate one-hot
weighted with a descending iota (`E - column`): the lowest tied column wins, exact, all in TILE layout.
Same-chip A/B (chip 4, layer 4, traced decode B=1, 20 steps, `_logs/mesh/perf_layer4_router_ab_{cumsum,rank}.log`):
10.74 ms/step -> 8.03 ms/step (-2.7 ms, -25 %); router sets identical to the reference path (32/32,
255/256, MoE PCC 0.999946). An argmax-based variant was also exact but slower (the index tensor's
layout conversions cost more than the cumsum). Knob `GLM53_ROUTER_TIE_BREAK=cumsum` keeps the old path.

## Root cause of the TP4 sparse-row gap (2026-10-20)

`test_tp4_sparse_prefill_layer3` first gave rows >= 2048 at 0.99777 (single chip 0.99990) while the
dense rows matched. Bisected with per-chip dumps of the sparse path (`_logs/mesh/sparse_tp4.pt` vs
`sparse_single.pt`): q, latent K/V and the sparse_sdpa kernel are bit-identical per head (the kernel is
exact with 16 real + 16 zero heads, `_logs/mesh/sparse_sdpa_heads_probe*.log`), fp32 all-reduce partials
change nothing, but the **index rows differ on chips 1-3** while chip 0 equals the single chip. The
selection intermediates (`select_tp4.pt`) show identical q / gates / pooled keys and identical finite
score VALUES on every chip; only the causal `-inf` mask differs: chip c scores as if its queries started
at `chunk_start + c * Sq`. `ttnn.experimental.indexer_score_dsa` treats every device of a mesh as a
sequence-parallel rank by default (`seq_shard_axes` unset = whole-mesh ring; documented in its nanobind
doc), so on the TP-replicated layout the non-origin chips saw FUTURE pools (non-causal selection ->
0.980 at the attention level, `test_tp4_attn_level.log`). The op-level repro on the (1,4) ring
(`_logs/mesh/indexer_score_mesh_probe2.log`): rows 0 visible pools per chip 512/576/640/704 with the
default or `[]`, 512/512/512/512 with `seq_shard_axes=[0]`. Fix: `TtDSAIndexer.select` passes
`seq_shard_axes=[0]` (the row axis, extent 1 on single-row meshes and on one chip) and refuses multi-row
meshes (those are the Phase-3 sequence-parallel engine, where rows ARE SP ranks). Note for the decode
path: the decode-mode selection happened to agree across chips because the not-yet-written future pools
score 0 against positive real scores; prefill fills the whole prompt before selecting.

## Next

1. E2E on the whole Galaxy: all 45 layers resident on the 8x4 (bfp8 experts, ~12 GB/chip), traced decode at 4 and
   32 users/row + TTFT through the sequence-parallel prefill (needs a board reset first: the chips of two blocks whose
   jobs were killed mid-flight have stuck ethernet cores, `llrt.cpp` "waiting for active ethernet core").
2. Per-row ownership for the SP prefill (KDA carry masks, scratch page tables for the DSA cache/index of non-owner rows).
3. Next decode targets from the perf report: mHC composite (1.9-2.2 ms/layer: single-core column slices with their
   tilize/untilize pairs, two single-core 32x16384x32 matmuls), KDA decode recurrence at 32 users/row (3.2 ms/layer),
   expert bfp4 gate/up under the eval gate, DSA small grids.
4. Sparse decode on multi-row meshes through the metadata-tensor variants (`chunk_start_idx_tensor` /
   `valid_end_tensor` of `indexer_score_dsa`); user-partitioned latent/index cache (design 5.3) and a paged/tiled kv
   read for `sparse_sdpa` (the 1 M untilize cost); hybrid vs `TTMoEDecode` A/B.
