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

1. B=32/row and the (1,4) TP prefill for layers 3/4 (running).
2. Full 8x4 mesh (needs all 32 chips free) with FABRIC_2D_TORUS_XY; A/B TP4 vs TP8.
3. Mesh prefill: ttKDA sequence-parallel over the rows + chunked DSA (after the chunked-prefill branch lands), user-partitioned latent/index cache, sparse decode through metadata tensors, 1 M-position cache test.
4. `tt-perf-report` of the mesh decode step (layer 4), hybrid vs `TTMoEDecode`, expert dtype under the eval gate.
