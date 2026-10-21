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

### 21:41 board incident and recovery

Every device open failed with `Timeout waiting for ARC msg request queue` (UMD power-state messages timing out on
all 32 chips); `tt-smi -s` still answered but its telemetry was stale (heartbeat identical across snapshots):
the chips' management firmware had stopped board-wide, right after a clean single-chip close and the next
full-mesh open. No process held a device. Host reboot by the user (~22:00); heartbeats advance again, the 8x4
sanity passed, queue relaunched (`_logs/mesh/sched_v7.sh`). Merged meanwhile: deterministic indexer stems for
chunked prefill (`prefill_linear`, 97642663bc2), partition_cols ROW_MAJOR path (586ad64479f), cache-hit shard
shape (661c25de72c), partitioned-test sizing (ab6e38d1e71).

### User-partitioned decode on the 8x4: first real exercise (23:30-23:45)

The 16 K needle run reached the decode capture and failed in the heads <-> users exchange:
`all_to_all_async_generic` grows `in_dim` and splits `out_dim` (the reverse of `GLMMesh.all_to_all_cols`'s
documented split/concat), so the 4-user dim landed on the tile-row position, where the op only accepts
multiples of 16. Fixed in the helper (cf302fd6434); the (2,4) partitioned test had never passed before, so
this path is being validated for the first time by the 16 K / 131 K runs on the fixed code.

KDA sequence-parallel prefill on 8 rows: a 32 K chunk needs 128 summary-owner cores (8 groups x 16 local heads) but the grid has 120, so the long-context runs use 16 K chunks (64 owners); the limit is chunk <= 24 K on the 8x4 with the default group size <= 20.

### 1 M needle run, stopped at a 512 K prefix for the optimization pass (2026-10-21 00:17-01:17)

Single slot per chip (no user partition, 1 user per row), 16 K chunks, 1 K scorer sub-blocks, composite mHC, 45 layers bfp8:
build 1026 s with the 1 M caches; decode capture at position 0 passed; prefill chunk times (s per 16 K tokens):

| prefix at chunk start | 0 | 16 K | 32 K | 48 K | 64 K | 128 K | 256 K | 384 K | 512 K |
|---|---|---|---|---|---|---|---|---|---|
| seconds | 136 (compiles) | 71 | 58 | 66 | 58 | 62 | 72 | 84 | 94 |

~52 s of prefix-independent work per chunk (~315 tok/s on 32 chips, ~0.3% of the compute roofline) plus ~10-12 s per 131 K of prefix (the whole-prefix latent copy per DSA layer). Projected 1 M TTFT ~98 min; the user stopped the run at chunk 32/64 to optimize first (prefix copy, slab-local attention, reference-path swaps, traced prefill).

## Slab-local sequence-parallel DSA prefill (2026-10-21, `arg/glm53_slabattn`)

v1 above gathered the whole chunk's hidden over the rows and ran `TtDSA.prefill_forward` on every row over all
T queries (the attention, the scorer / top-k and the per-chunk prefix copy 8x on 8 rows for 1x useful work), then
reduce-scattered identical copies. Now (`GLM53_SP_ATTN=slab`, the default; `v1` keeps the old path as the A/B
reference) every row attends only its own slab's T/rows queries against the full prefix (`TtDSA._prefill_slab`):

* stems on the local slab (q_a / q_b, kv_a, the indexer's k / gate); only the chunk's latent `c` [1,1,T,512] and
  the index stems k / g [1,1,T,128] are all-gathered over the rows (bf16, instead of the hidden [1,1,T,4096]),
  so the replicated latent cache and pooled-key index still receive the whole chunk exactly as before;
* SPMD: every row runs the same ops with the same shapes; what differs per row lives in row-sharded constant
  tensors built from the chunk geometry (host ints, cached): the dense SDPA's Q offset (the op's
  `windowed_q_token_offset_tensor`, start + r*T_loc: bit-equal to the plain causal kernel on every row), the
  scorer's causal bounds and tail ids, and the dense/sparse row select of a mixed chunk;
* sparse rows through the shared `indexer_score_dsa` (the merged prefill_1m_cost.md path): its per-row offset is
  chunk_start + r*Sq with Sq = the SUB-BLOCK rows, not the slab stride, so for contiguous slabs the chunk start
  is max_r(row_starts[r] - r*Sq) (a superset for every row, exact on the last) and the exact per-row bounds go
  into `_topk_bounded` row-sharded (`TtDSAIndexer.select(row_starts=...)`); then sparse_sdpa and o_proj on the
  slab only, output = the slab directly (no reduce-scatter, no 1/rows);
* row classes by absolute position: an all-dense or all-sparse chunk runs only that path per slab. A mixed chunk
  (start < 2048 < start + T) runs a uniform superset: the dense SDPA over the first n_d = min(T_loc, split) slab
  rows on every row (an all-sparse row is pointed at the last dense block, so its throw-away rows cost one dense
  block, never the whole prefix) and the sparse path over all T_loc slab rows (dense-position rows score at the
  clamped position 2048, a valid full prefix), then a row-sharded [n_d, 1] mask keeps the dense result below
  2048 (`ttnn.where`). Only the first chunk of a prompt is mixed;
* `_sparse_attention` (both prefill paths, `GLM53_SPARSE_A2A=1` default): the d_p `_sparse_mla` exchange
  instead of padding the 16 local heads to 32 for sparse_sdpa -- one all-to-all over the columns hands column c
  every head of sequence quarter c, the index rows are `mesh_partition`-ed alike, the attention runs 64 heads
  on a quarter of the rows, and the inverse exchange restores the head shard (sub-blocks with
  n % (32 * cols) != 0 keep the padded form);
* `GLM53_PREFILL_STEMS=fixed` (default): q_a, kv_a, q_b and the indexer's q projection go through
  `prefill_linear` (fixed K blocking in row blocks) like the k / gate stems already did, so a row's stems are the
  same whatever the chunk or slab length (the default dispatch picks its blocking per M: one-ulp differences
  between a 4096- and an 8192-row call flipped near-tie top-k selections on 2 of 14336 rows and 2 latent
  elements at 16 K; `default` keeps the old dispatch);
* the traced prefill (`slot`) stays on v1 (all-dense, <= 2048 rows; the slab path's first-use constants are
  not validated under trace capture); the KDA layers are untouched.

Tests: `tests/test_mesh_slab_prefill.py` on the (2,4) block (`_logs/agents/slabattn/`):

| test (slab slot 1 vs v1 slot 0, same rows) | result |
|---|---|
| `test_slab_primitives` | windowed SDPA with the row-sharded offset tensor bit-equal to the plain causal kernel on both rows; `where` with a row-sharded [n,1] mask exact; narrow gathers exact (`01_primitives.log`) |
| `test_dsa_attention_slab_r4097` (layer-3 attention, R4097, chunkings [4097] (a mixed slab: 2048 dense + 32 sparse rows), [1024 x4, 1], [2048, 2048, 1], [1024, 2048, 1025]) | 4097/4097 rows bit-exact in every chunking, 2049/2049 sparse selections identical, latent / pools / ring equal on all 8 chips (`05_attn_r4097_merged.log`) |
| `test_dsa_layer_slab_r4097` (whole layer 3 through FunctionalDecoder, [4097] and [2048, 2048, 1]) | pcc vs HF 0.999937 for both paths, 4097/4097 rows bit-exact; decode at 4097 / 4098 on both slots of both rows 0.999974 / 0.999967 vs HF, slot 0 vs 1 = 1.000000 (`03_layer_r4097.log`) |
| `test_dsa_attention_slab_long[16384]` (16384 real rows of R16385, [16384] and [8192, 8192]) | 16384/16384 rows bit-exact, 14336/14336 selections identical, state equal (`06_long_16k_fixedstems.log`; with the default stems 16380 / 15733 bit-exact rows and 2 selection flips, the known M-dependence) |
| `test_dsa_attention_slab_long[65536|131072]` (the same 16384 rows at the end of a synthetically seeded 64 K / 131 K prefix) | 16384/16384 rows bit-exact, selections identical, state equal in both chunkings, with and without the sparse a2a (`08_long_64k_131k.log`) |

Timing (`07_long_16k_64k_131k.log`, `08_long_64k_131k.log`: layer-3 attention, (2,4) block, warm wall per chunk with
no index recorder and no stage timer; v1 = the merged gathered-chunk path with the same shared scorer and a2a):

| prefix / chunk | class | v1 warm | slab-local warm | speedup | sparse a2a off: v1 / slab |
|---|---|---|---|---|---|
| 16 K: [0, 16384) one chunk | mixed (2048 dense rows) | 159 ms | 101 ms | x1.58 | 204 / 127 ms |
| 16 K: [0, 8192) | mixed | 80 ms | 58 ms | x1.38 | 101 / 71 ms |
| 16 K: [8192, 16384) | sparse | 91 ms | 43 ms | x2.11 | 109 / 57 ms |
| 64 K: [49152, 65536) one chunk | sparse | 219 ms | 136 ms | x1.61 | 234 / 147 ms |
| 64 K: two 8192 chunks | sparse | 94 / 92 ms | 48 / 49 ms | x1.95 / x1.87 | 128 / 61, 118 / 66 ms |
| 131 K: [114688, 131072) one chunk | sparse | 207 ms | 110 ms | x1.88 | 272 / 153 ms |
| 131 K: two 8192 chunks | sparse | 107 / 117 ms | 71 / 57 ms | x1.50 / x2.03 | 139 / 69, 138 / 72 ms |

The 16 K chunk at a 131 K prefix, one DSA layer: 0.11 s on the block against the merged single-chip baseline of
0.59 s (prefill_1m_cost.md, chip 0: scorer 207 ms, top-k 58 ms, sparse_sdpa 106 ms) and 0.21 s for v1 on the same
block. Two rows give the expected ~2x on all-sparse chunks (the per-row work is the slab's); the mixed first chunk
pays the uniform superset (one dense block on both rows, the sparse path over the whole slab: x1.4-1.6). The
stage-synchronized breakdown of the slab pass at 131 K (sums above the wall because of the per-stage syncs):
top-k group 85 ms (the [2048, 32 K-pool] score untilize + bounded top-k + id expansion per sub-block, 4 sub-blocks
per row), sparse_out 30, index queries 24, scorer 22, absorb 20, the two a2a legs 19 + 16, sparse_sdpa 17, o_proj 10,
cache fill + index 8, stems 7, slab gathers 3. The sparse a2a cuts sparse_sdpa from 45 / 23 ms to 13 / 6 ms per
8192 / 4096-row slab at 16 K (64 heads on a quarter of the rows instead of 32 padded heads on all rows) for
~5 ms of exchange per sub-block. The next terms are the scorer + top-k group (replicated over the columns: the
same heads-to-sequence split would divide it by 4) and the per-sub-block fixed costs (constants, zeros, slices).

All-dense chunks, the TTFT shapes (`test_dsa_attention_slab_dense`, `11_dense_short.log`: layer-3 attention, one chunk
of S tokens = S/2 rows per slab, warm minimum of 3, bit-exact in both cases):

| S | v1 warm | slab-local warm | slab stages (stage-synchronized) |
|---|---|---|---|
| 128 | 3.0 ms | 2.7 ms | stems 0.61, cache fill + index 0.92, dense SDPA 0.97, o_proj 0.35, the three gathers 0.40 |
| 1024 | 6.9 ms | 5.7 ms | stems 2.98, cache fill + index 5.84, dense SDPA 3.64, o_proj 2.23, gathers 2.91 |

TTFT, resident model layers 0-21 on the (2,4) block, 128-token prompt (`test_mesh_model.py -k ttft`, S=128): first
greedy token 21097 == the host head on the HF rows with both paths, chain pcc vs HF rows 0.964823 identical; warm
TTFT 365 ms slab-local (`12_ttft_22L_slab_warmcache.log`, cold 1.6 s) against 399 ms v1
(`10_ttft_22L_v1.log`, cold 7.1 s), both on the warm (2,4) weight cache.

Not done / next: the scorer + top-k group is still replicated over the columns (the heads-to-sequence split would
divide it by 4); the mixed first chunk's uniform superset costs one dense block per row plus the sparse path over
the dense rows (a caller splitting the first chunk at 2048 avoids it); the traced prefill keeps v1; the
single-chip `tests/test_select_rows.py` check (a mesh row's selections == the single-chip ids) was not re-run here
(no single-chip grant): the block tests compare the slab rows with the merged v1 rows, which that test validated
against the single-chip ids by emulation. Logs: `_logs/agents/slabattn/`.

### Multimodal on the (2,4) block (2026-10-21 03:52)

45 layers bfp4, image + question (286 tokens, 256 image tokens): answer "The image contains a red circle, a blue square, a green triangle, and the word \"HELLO\"." (first token == HF); TTFT 1201 ms (tower 90 ms); decode 145 ms/step = 6.9 tok/s/user over 2 slots.

## 1 M end to end on the 8x4 with the regions prefill path (2026-10-21 04:30-06:00)

The MoE "regions" prefill path (`doc/mesh/prefill_profile.md`; `GLM53_PREFILL_EXPERTS=regions`, merged a052cf309d6)
replaced the 32-token sparse_matmul blocks for the long runs. Two memory limits of the 1 M geometry surfaced on the
first 8x4 runs, both fixed in the model (no kernels):

| run | failure | fix |
|---|---|---|
| v14 1 M, chunk 1 | OOM: the region buffer `[T*k + 72*32, 4096]` bf16 of a 16 K chunk = 1.09 GB next to the 1 M caches (~1.5 GB free per chip) | `GLM53_REGIONS_T` row sub-chunks (50e1f58ae22; the 1 M runs use 2048; the expert weights are re-read per sub-chunk) |
| v15 1 M, chunk 64 (the ragged last one) | OOM in `ttnn.slice`: the index scorer's score block `[QBLOCK 1024, prefix/4 pools]` bf16 = 536 MB at a 1.047 M prefix, free DRAM fragmented (largest block 59 MB per bank) | `GLM53_PREFILL_QBLOCK=512` for the 1 M runs |

Chunk times of v15/v16 (45 layers, bfp8 experts, 16 K chunks, 8 rows): 7.0-7.6 s flat to 512 K, 9.2-9.4 s from
512 K to 1 M (v15, QBLOCK 1024), 7.0-7.3 s with QBLOCK 512; the first chunk and the ragged last chunk carry the
kernel compiles (15-31 s). The 03:00 run with the blocks path took 57-94 s per chunk over the same range.

### Needle ladder (single user, needle at depth 0.5, 32 greedy tokens; regions in 2048 sub-chunks, QBLOCK 512)

| ISL (tokens) | chunks | TTFT | prefill tok/s | first token | needle | vs HF CPU reference |
|---|---|---|---|---|---|---|
| 16359 | 1 | 41.8 s (cold) | 391 | 15209 'amber' | found | 8/8 greedy tokens identical (NEEDLE_16384) |
| 65493 | 4 | 37.3 s | 1755 | 'amber' | found | 8/8 identical (NEEDLE_65536) |
| 130839 | 8 | 73.0 s | 1791 | 'amber' | found | (no reference) |
| 261813 | 16 | 174.3 s | 1502 | 'amber' | found | - |
| 525170 | 32 | 319.7 s | 1643 | 785 'The' | MISSING | - |
| 1047526 | 64 | 497.2 s | 2107 | 785 'The' | MISSING | - |

The 512 K text at 32 tokens is an analysis monologue that quotes the question verbatim from the end of the 525 K-token
prompt; with `GLM53_LONG_GEN=128` the same run (depth 0.5, probe 06:15) retrieves the key AND its position:

> I need to find the secret passphrase embedded in the text. Let me search for it. Looking through the text, I see:
> "The secret passphrase for the vault is 'amber-falcon-7391'." This appears after day 12275 in the log. [...] So the
> answer is: amber-falcon-7391</think>amber-falcon-7391

(the first filler half has 12275 sentences, the needle follows it). So the stack retrieves across 262 K tokens; from
512 K on the model re-opens a reasoning block before answering although the prompt closes `<think></think>` (the chat
template's "Reasoning Effort: Max" system line), while up to 256 K it answers directly. Needle runs at >= 512 K need
`GLM53_LONG_GEN >= 128`.

### 1 M retrieval probes (160 generated tokens, regions in 2048 sub-chunks, QBLOCK 512)

| needle depth | distance to the question | first token | result |
|---|---|---|---|
| 0.98 | ~20 K tokens | 'amber' | found, direct answer: "amber-falcon-7391</think>The secret passphrase for the vault is 'amber-falcon-7391'." (TTFT 530 s, 1977 tok/s) |
| 0.75 | ~262 K tokens | 'amber' | found, direct answer (TTFT 507 s, 2066 tok/s) |
| 0.5 | ~523 K tokens | 'The' | MISSING: a degenerate filler loop ("On day 24178 the survey team measured a bakery and logged 878 entries ... lighthouse 729 ... tram depot 834 ..."), no task awareness (TTFT 458 s, 2288 tok/s) |

So at 1 M the question is attended and retrieval works at 20 K and 262 K distances (the 262 K distance is the one that
works at 512 K too); only the ~523 K distance fails, and it fails as a model-style collapse (filler continuation),
not as corruption. The selection chain is verified at the 1 M width independently: `tests/custom_ops/
test_wide_pool_selection.py` plants keys at pools 100 ... 262143 (both sides of 65536 and 131072) and every planted
pool is the scorer's row maximum and inside the bounded top-512 at W = 131072 and 262144 (worst top-512 overlap with
torch over the device scores 0.973 / 0.967, bf16 ties). A 1 M HF reference is out of reach on the host (the 64 K one
took 2385 s of CPU prefill per 16 K), so whether a 523 K-token single-sentence needle is a model limit of the
top-2048 sparse attention or a residual precision effect stays open; the next step is a hidden-state comparison
against HF on a shorter prefix that still crosses the chunk carries, or a multi-needle prompt.

Decode in these runs goes through the regions path (its per-expert lists replace the stacked expert tensors once
built): 4.2-4.6 tok/s per user at ISL 64 K-1 M, below the stream path (6.7 tok/s at 16 K on a block). Keeping both
forms costs 9.5 GB per chip (does not fit beside the 1 M caches); the fix is to point the decode stream op at the
per-expert buffers (a base-address table), queued after the retrieval work.

### Secondary stages on the 8x4 (v17, 2026-10-21 07:25-08:30; 45 layers, bfp8 experts, 4 users per row = 32 users)

Traced prefill (all-wrap mHC, fresh model): a 128-token prompt replays in **353 ms** (eager warm 1486 ms, cold 16.9 s;
x4.2), bit-identical across replays, first token == HF, chain PCC vs the HF rows 0.944.

mHC / expert-path bisection (`test_mesh_model.py -k chain`: 16-step decode chain vs HF prefill rows, then a 128-token
TTFT; every variant matches the greedy ids):

| variant | chain PCC min / median / max | traced decode step | tok/s per user (aggregate) | TTFT 128 tokens warm | TTFT PCC |
|---|---|---|---|---|---|
| fast mHC, exact pre/post (POST_KERNEL=2), prefill wrap | 0.751 / 0.812 / 0.827 | 75.5 ms | 13.2 (424) | 869 ms | 0.861 |
| + KDA fused kernel off | 0.798 / 0.831 / 0.866 | 94.5 ms | 10.6 (339) | 807 ms | 0.861 |
| fast mHC + dense expert layout | 0.745 / 0.789 / 0.805 | 95.2 ms | 10.5 (336) | 794 ms | 0.896 |
| all wrap (composite mHC everywhere) | 0.763 / 0.820 / 0.868 | 144.1 ms | 6.9 (222) | 1325 ms | 0.885 |

Reading: the mHC implementation does not set the 45-layer chain PCC (0.75-0.80 min in every variant); the fast
kernels halve the decode step (75.5 vs 144.1 ms) at equal accuracy, the fused KDA decode kernel is worth 19 ms per
step, the dense expert layout costs 20 ms at 4 users per row. The TTFT PCCs measured after the decode chain
(0.861-0.896) sit below the fresh-model 0.944 of the traced stage: a test-ordering effect to check (slot / KDA state
between the chain and the TTFT prompt), not a model property. Decision: fast mHC with the exact pre/post kernels
for decode, the composite wrap for the prefill rows; the needle runs re-graded vs HF with that config (v18) before
it becomes the code default (today's defaults: GLM53_MHC_IMPL=wrap, POST_KERNEL=0).

### mHC defaults flipped (2026-10-21 09:30, a641b9041e8) and the EP32 8x4 verdict

v18 re-ran the needle runs on the 8x4 with the fast configuration (`GLM53_MHC_IMPL=fast`, `POST_KERNEL=2`, exact
pre/post, `PREFILL_IMPL=wrap`; regions prefill in 2048 sub-chunks, QBLOCK 512):

| ISL | TTFT | prefill tok/s | greedy vs HF | decode (through the regions path) |
|---|---|---|---|---|
| 16359 | 11.8 s (one chunk, kernels cached) | 1381 | 8/8 identical | 164.5 ms = 6.1 tok/s per user |
| 65493 | 31.8 s (37.3 s with wrap) | 2060 | 8/8 identical | 148.3 ms = 6.7 tok/s per user |

So the fast kernels are the code default now (decode step 75.5 vs 144.1 ms at 32 users at the same chain PCC;
`GLM53_MHC_IMPL=wrap` keeps the composite everywhere for A/Bs). `GLM53_REGIONS_KEEP_STACKED=1` (5adf2791f3b) keeps the
stacked expert tensors next to the regions lists so decode stays on the stream op (second copy of the expert weights:
fits on the 8x4 up to ~512 K of context; validation in v20).

EP32 on the full 8x4 (`GLM53_MOE_LAYOUT=ep`, FABRIC_1D sanity passed): both e2e runs failed in the decode warm-up with an
L1 clash, "statically allocated circular buffers in program N clash with L1 buffers on core range [0-0 - 11-1 / 7-0]"
(256 users: top op `paged_update_cache` through the mHC forward; 4 users per row: `custom_ops/expert_stream/op.py`
`__call__` from `TtEPExperts.run_experts`). EP=8 on a (2,4) block passed earlier; the 32-way exchange's larger per-chip
token buffers overflow the L1 plan of the EP decode graph. Open item for the EP32 track; the hybrid layout is unaffected.
The EP weight conversion is cold on the first run (43 s per layer, 33 min), cached afterwards.

### Fresh-model TTFT and the keep-stacked decode (v19 / v20, 09:40-09:55)

A fresh-model `-k ttft` with the fast configuration gives chain PCC 0.901 on the 128-token prompt (fresh wrap 0.944;
fast after the chain test's decode chain 0.861): on 8 rows that prompt is 32 rows per chip, so `TtMHCSplit` serves it
through the fast decode-tile path (`PREFILL_IMPL=wrap` serves prompts above 32 rows per chip), which costs ~0.04 of
PCC on this short prompt and the chain ordering the rest; long prompts take the wrap path (16 K / 64 K HF-identical).

`GLM53_REGIONS_KEEP_STACKED=1`: block T2 3/3 with decode through the stacked path (PCC 0.99995-0.99996, above the
regions-kernel decode's 0.99986-0.99999); on the 8x4 both expert forms fit at 64 K, needle found and 8/8 identical to
HF, decode 165.6 ms per step with wrap mHC vs 215.9 ms through the regions kernel under the same wrap config
(the fast mHC through regions gave 148.3 ms; fast + stacked is the combination to measure next, ~100 ms expected at one
user per row).

### Decode with the fast mHC and the stacked expert path at 64 K (v22, 2026-10-21 13:10-13:35)

`GLM53_REGIONS_KEEP_STACKED=1` with the fast-mHC defaults, regions prefill, 64 K needle on the 8x4 (both results
8/8 identical to HF, needle found):

| users per row (total) | TTFT 65493 tokens | decode step | tok/s per user (aggregate) |
|---|---|---|---|
| 1 (8) | 31.8 s (2061 tok/s) | 101.8 ms | 9.8 (79) |
| 4 (32) | 36.2 s (1811 tok/s) | 99.0 ms | 10.1 (323) |

Against the morning's 215.9 ms (wrap mHC through the regions kernel) at one user per row. At ISL 16 the 32-user
step is 75.5 ms, so the 64 K context adds ~24 ms per step (sparse attention + index scoring); the 33 tok/s per
user target at 32 users (1 M ISL) is 3.3x away, the 19 tok/s at 256 users (121 ms at ISL 16) 2.3x: the
optimization phase's starting point.

## Optimization phase: per-op decode profile with the e2e knobs (2026-10-21 13:36-13:45, rows 4-5 block, tracy)

`tests/test_perf_mesh.py`, one traced decode step of one layer, device time of chip (0,0); the block's expert slices
are 1024-wide (4x the 8x4's 256), so the expert terms are ~4x the 8x4's; everything else transfers. Reports:
`_logs/mesh/report_prof2_{l4_b4_stream,l3_b4_stream,l4_b32_dense}/summary.md` (+ the op lists `dev0_r0c0_decode_ops.csv`).

| point | step | ops | mHC | router (+routing CCL) | experts | attention | CCL composites |
|---|---|---|---|---|---|---|---|
| layer 4 KDA+MoE, 4 users/row, stream op | 2.016 ms | 90 | 474 us / 34 ops | 168 us / 17 ops (+~80 us of broadcasts and pads) | 382 + 455 us (2 generic ops) | 229 us / 7 ops | 297 us / 23 ops |
| layer 3 DSA+MoE, 4 users/row, stream op | 2.398 ms | 133 | 475 / 34 | 168 / 17 | 440 + 441 | 609 us / 50 ops (dense DSA decode composite) | 255 / 23 |
| layer 4, 32 users/row, dense layout | 4.512 ms | 73 | 473 / 34 | 160 / 15 | 1044 + 1038 + 942 us (64 x 4096 x 73728 matmuls x2, 64 x 73728 x 4096) | 402 / 7 | 126 / 7 |

Each mHC pre-block is 16 ops of ~203 us: `MhcSplitSinkhorn` 52 us and a `LayerNorm` 46 us both on ONE core, two
tiny matmuls, reduces, transposes and pads; the post-mix v2 kernel is 33 us. The router is ~28 single-core-heavy ops
(`TopK` 49 us and `Gather` 33 us on one core, pads, untilize/tilize pairs, two column `AllBroadcast`s of the routing).
The dense expert matmuls at 32 users per row run at ~20 % of the bfp8 peak (14.5 GFLOP per chip per layer on the 8x4 in
~750 us). Budget to the targets (0.67 ms per layer at 32 users, 1.17 ms at 256): fused pre-block (A/B below), fused
router op (agent `routerop`), the dense-matmul program config, then the CCL pairs (RS+AG per all-reduce) and the DSA
decode composite.

## The serving envelope: users x context from the DRAM headroom (2026-10-21, `tt/memory_plan.py`)

The user count is not a fixed target: after the weights, the per-chip DRAM left is the budget for
`users x (ISL x cache_per_token + fixed_per_user)` plus a prefill working set. `tt/memory_plan.py` computes it from the
config (host side, `MemoryPlan.estimate`) or from the allocator after the weights are built (`MemoryPlan.from_device`);
`tests/test_memory_plan.py` pins it to the observed outcomes (the 1 M single-slot run fits, the stacked expert copy at
1 M does not, the partitioned cache at 1 M fits only without the SP-prefill scratch slot). 8x4, bfp8 experts:
weights 11.7 GiB per chip, cache 11.69 KiB per user per token (11 DSA layers x 512 bf16 latent + pooled index keys),
35 MiB of KDA state per user, headroom at 1 M 17.1 GiB (trace region 512 MB, regions buffers, score block, the
per-chunk whole-prefix latent copy of ISL x 576 bf16).

| ISL | replicated cache (today's long runs): users per row / on the mesh | partitioned + scratch slot (today) | partitioned, no scratch slot (heads -> users prefill, `mu1m`) | replicated + keep_stacked (stream decode) |
|---|---|---|---|---|
| 16384 | 84 / 672 | 332 / 2656 | 336 / 2688 | 41 / 328 |
| 65536 | 23 / 184 | 88 / 704 | 92 / 736 | 11 / 88 |
| 131072 | 12 / 96 | 44 / 352 | 48 / 384 | 5 / 40 |
| 262144 | 6 / 48 | 20 / 160 | 24 / 192 | 2 / 16 |
| 524288 | 3 / 24 | 8 / 64 | 12 / 96 | 1 / 8 |
| 1048576 | 1 / 8 | 0 | 4 / 32 | 0 |

Reading: the partitioned cache without the scratch slot is the envelope (32 users at 1 M, 96 at 512 K, 192 at 256 K);
the stacked-expert decode copy is a 9.5 GiB trade against it (fine up to ~512 K at one user per row, or 11 users per
row at 64 K). The decode-speed work therefore has two regimes: few users at very long context (the sparse DSA decode
and the per-layer fixed costs: mHC, router, CCL) and many users at short context (expert compute: EP32 dispatch).
