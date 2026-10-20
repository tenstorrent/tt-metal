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
| **8x4 Galaxy, all 45 layers, bfp8 experts, rank router** | 0-44 | 4 (32 users) | 0.850-0.898 per user (16 tokens; chaos regime, see the chain note; rows differ because the 8-row bf16 reduce-scatter associates the expert partials differently per destination row) | 75 % of users at position 15 | **203.35 ms -> 4.9 tok/s per user, 157 tok/s aggregate** | 4.52 ms | `test_8x4_45L_bfp8_B4.log` (32 min incl. ~27 min weight loading) |
| **8x4 Galaxy, all 45 layers, bfp8, rank router** | 0-44 | 32 (256 users) | 0.850-0.898 per user | 75 % | **426.5 ms -> 2.3 tok/s per user, 600 tok/s aggregate** | 9.48 ms | `test_8x4_45L_bfp8_B32_2.log` (trace region 512 MB: the 45-layer step at 32 users/row captures 216 MB of trace); TTFT warm 2548 ms |

**TTFT on the 8x4** (same run): 128-token prompt through the sequence-parallel prefill v1 of all 45 layers + head:
warm 2587 ms, cold 36.3 s (compiles); first greedy token 102647 == host head. Slower than the 2x4 block (1301 ms)
because v1 splits 128 tokens into 16-token slabs per row and is latency-bound (eager, ~55 ms per layer of small
ops: full-chunk gathers for the DSA layers, 8-row expert gathers); a prefill-specific pass (batched rows, fewer
collectives, trace) is the next TTFT step.

**TTFT, full model** (same run, `_ttft`): 128-token prompt through the sequence-parallel prefill of all 45 layers +
head: warm 1301 ms, cold 2660 ms (first-call compiles); first greedy token 102647 == host head on the HF rows.

**Chain note:** the chain PCC vs HF after many layers drops (6 layers 0.9993, 24 layers bfp8 0.876, 45 layers bfp4
0.80-0.83) while the greedy token stays identical; whether this is precision accumulation over the 42 MoE layers
(near-tie expert flips compound) or a chain-level defect is being separated with the single-chip streaming full
model (`arg/glm53_fullmodel`, true prefill path, per-layer chain PCC).

**Resolved (control):** the HF bf16 chain is itself this sensitive. `reference/hf_layer_reference.py --perturb 1e-3`
(relative Gaussian noise on the layer-0 input, 5.0e-4 relative rms after bf16 rounding) chained through the same
45 layers vs the unperturbed `L45_S128` dump (`tests/test_chain_sensitivity.py`, `_logs/mesh/chain_sensitivity.log`):
prefill-row PCC 0.99994 (layer 3), 0.9984 (12), 0.9925 (18), 0.971 (21), 0.930 (24), 0.880 (27), 0.837 (30),
0.829 (33), 0.908 (36), 0.905 (42), 0.934 (44); per-row minima down to 0.55. The TT single-chip streaming chain
(bfp8 experts, true prefill: 0.998 at layer 10, 0.958 at 20, 0.69 at 30, 0.86 at 44) and the mesh chains follow the
same curve one notch lower (its per-layer error is ~1e-3, twice the perturbation). Chain PCC past ~20 layers is
therefore not a correctness metric for this model; token-level accuracy (greedy / teacher-forced top-1, top-5) is.

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

## Per-layer profile at the real 8x4 layout (2026-10-20, `tests/test_perf_mesh.py` GLM53_PERF_MESH_SHAPE=8,4, rank router)

Chip (0,0) device time of one traced decode step (reports `_logs/mesh/report_perf8x4_l{4,3}_b{4,32}/summary.md`,
per-chip tables in the same dirs; device ids row-major `[0,4,12,8,1,5,13,9,...]`):

| layer | users/row | step | mHC | experts local (sparse_matmul) | router | attention | CCL composites (kernels) |
|---|---|---|---|---|---|---|---|
| 4 KDA+MoE | 4 | 5.25 ms / 524 ops | 1.89 ms (36 %, 220 ops) | 1.10 ms (0.86: gate|up [32x4096x512] 733 us, down 130 us) | 0.70 ms (139 ops) | 0.89 ms (75 ops) | 0.48 ms, 59 ops (0.24 ms, 7 kernels) |
| 4 | 32 | 15.54 ms | 2.23 ms | 8.48 ms (6.48: 8 blocks x gate|up ~700 us each: the active experts' weights are re-streamed per 32-token block) | 0.54 ms | 3.25 ms (KDA recurrence at 32 users) | 0.89 ms (ReduceScatter 653 us) |
| 3 DSA+MoE | 4 | 5.02 ms | 1.88 ms | 1.00 ms (0.76) | 0.70 ms | 0.74 ms | 0.55 ms (0.32) |
| 3 | 32 | 12.70 ms | 2.22 ms | 8.05 ms (6.05) | 0.54 ms | 0.91 ms | 0.84 ms |

Reading: against the design budget (~0.7 ms per MoE layer at 4 users/row, ~1.2 ms at 32), every term is a
composite-overhead or kernel-shape problem, not a bandwidth one: (1) mHC = 220 small ops; (2) the gate|up
expert matmul at N = 512 runs on 16 cores (output-width split) and, at 32 users/row, is repeated per 32-token
block; (3) the router = 139 ops; (4) the KDA decode recurrence composite; (5) the generic collectives expand to
59 ops at 4 users/row (AllBroadcast + pads for the non-tile-aligned row gather). Work items 1-4 run in
parallel on branches `arg/glm53_optmhc`, `arg/glm53_optexperts`, `arg/glm53_optkda` plus the router top-k
variant in this branch.

## Optimization 2: routed-expert kernels (2026-10-20, `expert_kernel_sweep.md`)

Single-chip sweep of `ttnn.sparse_matmul` at the 8x4 per-chip shapes (`tests/test_expert_kernel_perf.py`, raw
tables in `expert_kernel_sweep_tables.md`): the N = 512 gate|up call is latency-bound by its 4-tile K blocks on
16 cores (1092 -> 369 us per 32 active experts with 64-tile blocks, `sparse_matmul_config`), the down call wants
32 cores x 4 tiles (171 -> 144 us), and no sparse configuration handles 256 gathered tokens (one call 6.6 ms,
8 blocks ~9.5 ms). The dense layout (`TtRoutedExperts(dense=True)` / `GLM53_DENSE_EXPERTS=1`: one
[4096, 72x512] and one [72x256, 4096] matmul over all local experts, expert sum = K reduction, no slab / fill /
reduce) runs the whole expert block in 1.48 ms at 256 real tokens in one pass against 9.7 ms for today's 8 sparse
blocks (`tests/test_moe_block.py`, real layer-4 weights, PCC 0.99996 vs the current block), the recommended decode
path at >= 8 users/row; at <= 4 users/row the sparse path with the new config (1358 -> 705 us per block) stays.
Op-count pass (`expert_kernel_sweep.md` section 7): dense block 11 -> 6 device ops (two gate/up matmuls,
`clamped_silu_glu`, one routing-weight matmul from the router's bf16 matrix, the multiply, W2), 1475 -> 1231 us at
256 real tokens; sparse block 18 -> 13 device ops (`clamped_silu_glu`, one `permute` for the routing weights, no
typecasts), bit-identical output.
Kernel pass (`expert_kernel_sweep.md` section 8, evidence only, not adopted): a local tt-metal build of
`sparse_matmul` with batch groups, in0 reuse, an accumulate-over-experts down mode and no zero-fill reached 394 us
per 32-token block at 28 real active experts (gate|up 204 us at 60 % of DRAM peak, down 92.5 us at 66 %); reverted
under the "model-local generic ops only" rule and kept as the bound for the custom op.
Custom op (`custom_ops/expert_stream/`, section 9): the expert block as two `ttnn.generic_op` DRAM-bank-streaming
programs after `gpt_oss/tt/fused_decode/stream.py` (weights width-sharded over the 8 banks, bank-major
column-major; 96 gate|up cores with `clamped_silu_glu` in DST and the routing weight as a diagonal-tile matmul,
16 down cores accumulating the expert sum in fp32 DST and writing once), trace-safe (every kernel scans the routing
matrix on device, no host counts). Block at 4 users/row (28 real active): 683 -> 341 us (3 device ops, PCC
0.99997 vs the sparse block, 54 % of DRAM peak), `GLM53_STREAM_EXPERTS=1`; at 256 tokens the block form
re-streams the weights per 32-token block (2.41 ms vs 1.26 ms dense).
Phase 2 (section 10): the per-expert dispatch ops (`ExpertDispatch`: on-device gather of each expert's routed rows
from the row-major tokens, tilize, the same gate|up kernel; down projection per pass with the token sums gathered
in L1 and summed in fp32 DST), every active expert read once, three small ttnn ops + two generic ops; real routing
M = 256: 0.78 ms per layer per chip, 6 device ops, PCC 0.99995 vs the sparse block (dense 1.26 ms, blocks 2.41 ms), `GLM53_STREAM_DISPATCH=1` default above 32
tokens. The op is also generalized to the wider per-chip slices of the smaller meshes (I = 1024 on a (2,4) block:
4 column pairs per bank, 4 down readers; 2048 on one chip), 8x4 unchanged.
EP32 path (orchestration v2, `expert_kernel_sweep.md` section 11): `TtEPExperts` / `GLM53_MOE_LAYOUT=ep`: every
chip owns `E / D` full-width experts, the dispatch op in EP mode (routing from the router's top-k lists + the chip's
expert base, bfp8 or bfp4 gate|up), `all_to_all_dispatch` / `all_to_all_combine` along the rows on a mesh
(written, untested until a block is free); single-chip path validated vs the hybrid stream path on real layer-4
routing (`tests/test_moe_ep.py`): exact vs the hybrid stream path (PCC >= 0.99999), 9 full experts 297 us at 32 tokens /
590 us at 256 (3 of 9 active), 36 experts 1.38 / 2.07 ms; real-routing dispatched-row imbalance max/mean 2.1-2.4 on the 8x4,
1.3-1.5 on a (2,4) block.

## Optimization 1: router tie-break (2026-10-20)

The exact fp32 top-8 router's first-match tie-break used `ttnn.cumsum` (single-core, 330 us x 8 rounds =
2.64 ms per MoE layer, replicated on every chip). Replaced by a second max over the candidate one-hot
weighted with a descending iota (`E - column`): the lowest tied column wins, exact, all in TILE layout.
Same-chip A/B (chip 4, layer 4, traced decode B=1, 20 steps, `_logs/mesh/perf_layer4_router_ab_{cumsum,rank}.log`):
10.74 ms/step -> 8.03 ms/step (-2.7 ms, -25 %); router sets identical to the reference path (32/32,
255/256, MoE PCC 0.999946). An argmax-based variant was also exact but slower (the index tensor's
layout conversions cost more than the cumsum). Knob `GLM53_ROUTER_TIE_BREAK=cumsum` keeps the old path.
On the (2,4) block (`test_perf_mesh.py`, layer 4, 4 users/row, rows-6-7 block, `meshperf_router_ab_rows67_*.log`):
8.03 -> 5.34 ms/step (-34 %), i.e. ~-113 ms on the 318 ms 45-layer step above (42 MoE layers).

## Optimization 2: router as one fp32 top-k (2026-10-20)

`ttnn.topk` accepts fp32 inputs (stable bitonic network on Blackhole), so the exact top-8 is one op: topk over
the biased scores -> `ttnn.gather` of the unbiased scores at the winners -> renormalize x 2.5 -> `ttnn.scatter`
into the dense [T, E] routing matrix (bf16: scatter has no fp32 TILE path, and the expert block applied bf16
weights anyway). ~10 ops instead of ~90. Same-chip A/B chip 4, layer 4 traced decode B=1:
rank 8.03 -> topk 7.60 ms/step; routing identical to the reference composite (32/32, 255/256, dense weights
0.999872, MoE PCC 0.999946; `_logs/mesh/router_topk_test_moe_3.log`, `perf_layer4_router_ab3_*.log`). Default
now `GLM53_ROUTER_TIE_BREAK=topk` (`rank` / `cumsum` kept for A/B).

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
4. Sparse decode at full ISL: the traced multi-user path (per-user metadata tensors, pool gather into a compact
   cache, batched flash-MLA decode; no whole-prefix copy) and the user-partitioned cache design are in
   `sparse_1m.md` (track OPT_SPARSE_1M, branch `arg/glm53_sparse1m`); hybrid vs `TTMoEDecode` A/B.

## Optimization round 1 (2026-10-20): data-movement ops out, projection configs in

Directive: every data-movement op is merged into a single op or removed before any compute tuning (the
step has to end up weight-bandwidth bound; at 45 layers the 30 ms / 32-user target is ~0.67 ms = ~60-80
device ops per layer). Changes on `arg/glm53_mesh` (4d381ab55d9, 1630e1dc250, merge 49f6812fdce):

| item | before | after | evidence |
|---|---|---|---|
| row collectives (`GLMMesh.all_gather_rows/cols`, `reduce_scatter_rows`) | generic composites: on 8 rows all_broadcast + 8 untilize + concat + tilize (gather), 8 fill_pad + concat + rs (reduce-scatter) | one `*_async` op each (persistent TT_CCL semaphores), `GLM53_CCL=generic` keeps the composites | `tests/test_ccl_perf.py` on the 2x4 block: same per-call time (37 / 75 us at B=4); all_reduce stays generic (async 58 vs 53 us) |
| decode projection matmuls (16 per-chip shapes) | default dispatch, 30-240 GB/s of bfp8 weights | 1D `mcast_in0` configs on interleaved tensors, 200-330 GB/s (`tt/matmul_configs.py`) | `tests/test_matmul_sweep.py` chip 4: 959 -> 497 us summed, PCC identical; DRAM-sharded weights lose once the activation reshard is counted |
| DSA decode stem (q_a, kv_a, indexer k, pool gate, head weights) | 5 matmuls on 4-48 cores, ~170 us | one [4096 x 2336] bf16 matmul on 74 cores + tile-aligned slices | layer 3 decode / sparse decode 0.99995 / 0.9999+ (`test_decoder_layer`, `test_dsa_sparse`) |
| absorb batched matmuls | 65.8 / 106.4 us (16 / 8 cores) | one core per head: 24.1 / 23.2 us | `tests/test_bmm_sweep.py` |
| MLP activation | slice, slice, clamp, clamp, silu, mul | slice, slice, `ttnn.clamped_silu_glu` | layer 0 unchanged (0.99996) |
| router | 139 ops (rank) -> 13 (top-k) | 9: sigmoid fused into the gate matmul, sum -> rdiv -> mul, no host casts | identical top-8 sets 32/32, 255/256 (`test_moe`) |
| LM head | replicated 0.63 GB head on every chip, 5 chunk matmuls + concat | vocab split over the TP columns, one [4096 x 38720] matmul per chip + `all_gather_cols` | first token == host head (`test_mesh_model` 2x4, 4 and 6 layers) |
| experts (agent, `expert_kernel_sweep.md`) | sparse N=512 gate|up 1092 us (4-tile K blocks on 16 cores), 8 slab blocks at 32 users/row (9.7 ms) | sparse config from the sweep (369 us); dense layout `GLM53_DENSE_EXPERTS=1`: one [4096, 72x512] + one [72x256, 4096] matmul, no slab fill / expert-sum (1.48 ms per layer at 256 tokens) | `tests/test_moe_block.py` real layer-4 weights, PCC 0.99996 vs the sparse block |

2x4 block, chip (0,0), one traced decode step at 4 users/row after the first six items (reports
`_logs/mesh/report_perf2x4_r1_l{4,3}_b4/`): layer 4 368 ops / 4.61 ms device, layer 3 334 ops / 4.21 ms.
What is left is the mHC glue (Slice 80, UntilizeWithUnpadding 54, TilizeWithValPadding 49, Ternary 38 of
the 368), the KDA head split + recurrence composite, and the single-core input norms: the three agent
tracks (mHC fused pre/post blocks with the norm folded in, fused KDA decode, dense expert path op count).

## Optimization round 2 (2026-10-20): the three agent tracks merged

Merged on `arg/glm53_mesh` (experts 996ce99d9a7, KDA op 23ec6eb31fc, mHC e41b669e90e, KDA block default
b8d4f49aab3, exact-residual option a65c66d48c3; upstream code 3e9df80eb0d). The tree now needs the wt_mesh
C++ build (`ttnn.transformer.kda_decode`); see `doc/functional_decoder/README.md` (KDA block), `doc/mhc/README.md`
(TtMHCFast, exact option), `expert_kernel_sweep.md` (expert layouts).

| track | before | after (default path) | evidence |
|---|---|---|---|
| mHC (2 wraps per layer) | 222 ops, 1.96 ms (B=4) | 44 ops, 0.46 ms, input norms folded in, head collapse folded into the last layer's post-block | chip 1 Tracy; PCC 1.0000 vs composite (residual update rounds ~1e-3 rel rms: `GLM53_MHC_EXACT=3` is exact at +16 ops/wrap, off) |
| KDA decode | ~75 ops, 11.4 ms at 32 users (64 heads) | 6 ops (qkv, 2 gate matmuls, `kda_decode`, tilize, o_proj), 1.49 ms | chip 3; block vs composite 0.99996; mesh TP4 at 32 users/row 0.99999 median after the per-column gate-weight split |
| experts | sparse 8 slab blocks 9.7 ms at 256 tokens; block 18 ops | dense 6 ops 1.23 ms (>= 8 users/row, `GLM53_DENSE_EXPERTS=1`); sparse 13 ops 683 us per block | chip 2 `test_moe_block.py` PCC 0.9999; mesh 0.99992 |

2x4 block, chip (0,0), traced decode step at position 1024, sparse experts (the dense layout reads the
block's 4x larger per-chip slice and is only meaningful on the 8x4); reports `_logs/mesh/report_perf2x4_r3_*`:

| layer | users/row | round 1 | round 2 (merged) | what is left (ops) |
|---|---|---|---|---|
| 4 KDA+MoE | 4 | 368 ops / 4.61 ms | **112 ops / 2.34 ms** (sparse_matmul 0.75) | BinaryNg 22, Matmul 16, Slice 9, Untilize 9, Reduce 8, FillPad 7, FastReduceNC 5, Tilize 5 |
| 4 | 32 | - | **109 ops / 3.92 ms** (sparse_matmul 1.92) | |
| 3 DSA+MoE | 4 | 334 ops / 4.21 ms | **147 ops / 2.42 ms** (sparse_matmul 0.66) | + Slice 16, LayerNorm 5 (single core), ReshapeView 5, cache writes 8 |
| 3 | 32 | - | **144 ops / 3.95 ms** | |

Context of every number above: decode position 1024 (per-layer profiles) and ISL 16 (the e2e), both below the
2051-token dense/sparse switch; the design's floor (19 / 33 tok/s/user) is at 1 M ISL, where the index +
sparse-attention path (single-chip single-user 26.5 ms per layer today) has to come down to ~0.7 ms per layer:
track OPT_SPARSE_1M (`_agents/OPT_SPARSE_1M.md`). Next per-track items: fused exact fp32 mHC post-block kernel
(2 ops, exact), DSA decode glue (cache writes, pooling, head transposes, norms), sparse_matmul kernel efficiency
at low batch (38 % -> 70 % of DRAM peak).

## 2026-10-20 evening: first 8x4 run with the custom ops (hang, root-caused)

The 45-layer bfp8 e2e at 4 users/row (fast exact mHC, KDA block op, stream experts) hung in the decode-trace capture
on the 8x4: `tt-triage` on the live process showed the o_proj matmul (`[1,1,4,2048]`, trace 0) stuck on 9 of 32
chips, one core each, its in0-receiver ncrisc in `noc_async_atomic_barrier`, the row-mates waiting in the following
column reduce-scatter, and the `kda_decode` generic op as the previous op. The KDA reader/writer kernels exited with a
cross-core `noc_semaphore_inc` still in flight; the next kernel snapshots the NoC atomic counters at start, so the late
ack leaves them unequal forever. Fix 20eaa501a00: `kda_decode` and `mhc_pre_block` kernels end with
`noc_async_full_barrier()`. Also from this run: the 45-layer build costs ~60 s per layer of host weight conversion;
the e2e test and demo now build through the tensorbin cache (`GLM53_MESH_CACHE`, default `_cache/mesh`), first run
writes it.

### 8x4 decode after the kernel barrier fix (2026-10-20 19:38, run r4)

| config (45 layers, bfp8 experts, ISL 16) | step | tok/s per user | aggregate | chain PCC median | greedy id == host head |
|---|---|---|---|---|---|
| morning (composite ops), 4 users/row = 32 users | 203.35 ms | 4.9 | 157 | 0.868 | 75 % |
| r4 (fast exact mHC + KDA block + stream experts), 4 users/row = 32 users | 76.45 ms | 13.1 | 419 | 0.842 | 100 % |
| morning (composite ops), 32 users/row = 256 users | 426.54 ms | 2.3 | 600 | 0.868 | 75 % |
| r4, 32 users/row = 256 users | 121.56 ms | 8.2 | 2106 | 0.851 | see log |

TTFT at 128 tokens (r4): warm 919 ms (4/row) and 971 ms (32/row) vs 2587 ms in the morning, but the first token
differs from the host head (39030 vs 102647, chain PCC 0.82 vs 0.91): the prefill-side regression under the fast exact
mHC path; A/B with the composite wrap on the prefill rows queued (`GLM53_MHC_PREFILL_IMPL=wrap`).

Demo (`prefill_128-4users`, 8x4, 45 layers bfp8, 21:05): ISL 79-92, TTFT mean 2896 ms, decode 166.63 ms/step =
6.0 tok/s/user, 192 tok/s aggregate with the composite hybrid experts (the demo sets no expert policy); the e2e r4
step with stream experts is 76.45 ms, so the expert path is ~90 ms of the 4-users/row step. Outputs coherent.
