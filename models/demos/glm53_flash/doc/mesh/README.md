# GLM-5.3-Flash Phase 2: multi-chip decoder (work in progress, 2026-10-20)

## Status (2026-10-22 morning)

| area | state | where |
|---|---|---|
| layout | TP4 x DP8, hybrid experts (72 per chip, intermediate over the 8 rows), user-partitioned DSA cache (heads2users), prefill: blocks <= 4 K positions, regions above (auto default) | this doc; `tt/mesh_model.py` |
| accuracy | logits PCC 0.994 / top-1 90 % / top-5 94 % vs HF on natural text; 128-token first token == HF; needles 32/32 at 1 M, 64/64 at 512 K, 128/128 at 256 K, 256/256 at 128 K; greedy == HF at 16 K / 64 K | `tests/test_mesh_model_logits.py`, README |
| envelope | 32 x 1 M, 64 x 512 K, 128 x 256 K, 256 x 128 K users measured (1.1-2.0 GiB spare) | README table, `tt/memory_plan.py` |
| decode | 12.5 tok/s (batch 1) / 12.0 (32 users) / 8.7 (256); long context 6.7 (1 M) ... 5.0 (256 x 128 K); targets 60 / 33 / 19 | README; ledger `_agents/decode_ledger/` |
| TTFT | 265 ms traced at 128 tokens; 2.2-2.4 K tok/s chunked prefill (27 s at 64 K, 460 s at 1 M, per prompt) | README |
| open accuracy items | EP32 prefill first token wrong (experimental layout); slot-3 near-tie divergence in degenerate 1 M regime (parked) | ledger entries below |
| optimization queue | A3/A4 mHC norm + Sinkhorn, B7 TP all-reduce as gather + reduce, C1 rs combine for 4 users/row (agents, code only, 2026-10-22), A1 audited (`mhc_pre_block_audit.md`: no NOC hazard behind the 0.70, idle-core gather race fixed, mesh tests in the chain); B5 DSA decode state fusion code-complete (`GLM53_DSA_DECODE_STATE`, device A/B in the chain); MTP steps 3-5 code-complete (`tt/spec_decode.py`; device tests in the validation chain); prefill: incremental prefix copy + scorer fusion | `_agents/*/STATUS.md` |
| serving | vLLM generator drafted (no device yet) | `_agents/vllm_glm53/` |
| PP | design study: no decode gain at equal batch, ~1.3-1.5x users at 1 M and 2-4x prefill ceiling; second-phase layout | `pp_design.md` |


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

### EP32 L1 clash: root cause and fix (2026-10-21, ep32l1)

Not the 32-way exchange. `TtEPExperts.combine_rows` pre-zeroes the `all_to_all_combine(local_reduce)` output by
cloning a persistent zero constant `[K=8, T, 1, H]` bf16 ROW_MAJOR (`ttnn.zeros` is a host write, not allowed in a
trace), and that constant was built **in L1, once per MoE layer**. Interleaved L1 pages are 8 KB and round-robin
from bank 0, so every layer pinned 24 KB (32 users/row: 256 pages) / 8 KB (4 users/row: 32 pages) of the first
cores' L1 for the life of the model, and the region left for the next program's static circular buffers shrank by
that much per layer:

| users/row | per-layer L1 constant | clash in the log | arithmetic |
|---|---|---|---|
| 32 | 2 MB = 24 KB on cores 0-15 | `paged_update_cache` (dataflow buffers end at 1361056), L1 buffers down to 1351232 | 1572864 - 1351232 = 216.4 KB = 9 MoE layers x 24 KB: the 10th MoE layer's DSA update fails |
| 4 | 256 KB = 8 KB on cores 0-31 | gate|up dispatch op (CBs end at 1215872), L1 buffers down to 1207808 | 1572864 - 1207808 = 356.5 KB = 42 MoE layers x 8 KB + the op's own transients: the last layers fail |

The footprint depends on T (users per row) only, not on the EP group size; the (2,4) block had only ever run ONE EP
layer (`test_mesh_moe.py`), which is why "EP=8 passed". The hybrid layout has no such constant. Fix: the zero
constant lives in DRAM and is shared by every EP layer of the model (`GLMMesh.const`, one tensor per shape); the
op graph is unchanged (the per-call `ttnn.clone` into L1 now reads DRAM: 2 MB per layer at 32 users/row).
`test_mesh_moe.py` now asserts that the L1 high-water mark (the address of a fresh one-tile L1 probe, top-down
allocator) is identical before and after the MoE forward, so a lazily built per-layer L1 constant fails the
single-layer block test. Validation on the rows-6-7 (2,4) block (EP = 8, FABRIC_1D, bfp8 experts, today's defaults:
fast mHC; logs `_logs/agents/ep32l1/block_*.log`):

| stage | code | config | result |
|---|---|---|---|
| pre-fix repro | arg/glm53_mesh @8b79b5d9f9a | layers 0-13, 32 users/row | the SAME clash as the 8x4: `paged_update_cache` program 244, dataflow region ends at 1361056, L1 buffers down to 1277440, core range [0-0 - 11-1] |
| post-fix | arg/glm53_ep32l1 @835a23dc959 | layers 0-13, 32 users/row (64 users) | pass: chain PCC 0.992108 vs HF after layer 13 (greedy 100 %), decode 30.17 ms/step = 33.1 tok/s per user, 2121 tok/s aggregate; TTFT 128 tokens warm 272 ms, prefill PCC 0.997012, first token = host head |
| post-fix | same | layers 0-21, 4 users/row (8 users) | pass: chain PCC 0.942429 (hybrid 22L reference 0.942288), decode 31.61 ms/step = 31.6 tok/s per user, 253 tok/s aggregate; TTFT warm 313 ms, prefill PCC 0.963906 |
| post-fix | same | layers 0-21, 32 users/row (64 users) | pass (pre-fix: a clash after ~10 MoE layers): chain PCC 0.942429, decode 47.67 ms/step = 21.0 tok/s per user, 1342 tok/s aggregate; TTFT warm 429 ms, prefill PCC 0.963906 |
| `test_mesh_moe.py` ep | same | T = 2 / 32 per row | 2 passed: PCC 0.999940 / 0.999923; L1 high-water mark [before, iter 0, iter 1] = [1538048, 1537600, 1537600] (the 448 B = the model's TT_CCL semaphores, allocated once on the first collective), no L1-resident module state |
| `test_mesh_moe.py` hybrid (FABRIC_2D) | same | T = 2 / 32 per row | 2 passed: PCC 0.999939 / 0.999922; marks [1538048, 1534976, 1534976] / [1538048, 1535488, 1535488] (its collectives' semaphores, once), no L1-resident module state: the invariant holds for both layouts |

For scale, the hybrid 22-layer block runs of 2026-10-20 (composite mHC then) took 39.27 ms/step at 4 users/row and
92.0 ms at 32 (dense experts); the EP block decode at 32 users/row is 2.17 ms per layer against 2.16 at 14 layers,
so the per-layer cost is flat in the layer count. The EP conversion of a block layer is cold at 80-130 s (36 full
experts per chip), ~10-20 s from the `mesh2x4_ep` cache afterwards.

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
`tests/test_memory_plan.py` pins it to the observed outcomes (the 1 M single-slot run fits by ~60 MB, the stacked
expert copy at 1 M does not, the partitioned cache at 1 M fits only without the SP-prefill scratch slot, 4 users per
row at 64 K with both expert forms ran). Calibration: 28.85 GiB allocated after a 45-layer bfp8 build with one 1 M
slot per chip (the `mu1m` 8x4 run) -> 17.6 GiB of weights + fixed buffers per chip (9.4 GiB of bfp8 experts + 8.2 GiB
of dense TP4 weights, expert layouts, embedding / head shards, the traced-prefill slot, tables; attribution pending),
cache 11.69 KiB per user per token (11 DSA layers x 512 bf16 latent + pooled index keys), 35 MiB of KDA state per
user, working set = trace region + max(whole-prefix latent copy of ISL x 576 bf16, regions buffers + score block).

| ISL | replicated cache (today's long runs): users per row / on the mesh | partitioned + scratch slot | partitioned, no scratch slot (heads -> users prefill, merged) | replicated + keep_stacked (stream decode) |
|---|---|---|---|---|
| 16384 | 57 / 456 | 224 / 1792 | 228 / 1824 | 14 / 112 |
| 65536 | 16 / 128 | 60 / 480 | 64 / 512 | 4 / 32 |
| 131072 | 8 / 64 | 28 / 224 | 32 / 256 | 2 / 16 |
| 262144 | 4 / 32 | 12 / 96 | 16 / 128 | 1 / 8 |
| 524288 | 2 / 16 | 4 / 32 | 8 / 64 | 0 |
| 1048576 | 1 / 8 | 0 | 4 / 32 | 0 |

Reading: the partitioned cache without the scratch slot is the envelope (32 users at 1 M with ~0.6 GiB to spare, so the
per-user release discipline and a 256-query score block matter there; 64 at 512 K, 128 at 256 K); the stacked-expert
decode copy is a 9.4 GiB trade against it (4 users per row at 64 K). The decode-speed work therefore has two regimes:
few users at very long context (the sparse DSA decode and the per-layer fixed costs: mHC, router, CCL) and many users
at short context (expert compute: EP32 dispatch).

### Optimization-phase agents, first deliveries (2026-10-21 14:00-14:45)

| track | result | state |
|---|---|---|
| `routerop`: fused exact-fp32 router top-8 (`custom_ops/router_topk`, one program, one token row per core) | single chip, layer 4, 32 users: the 14-op / 142 us composite -> 1 op / 13.2 us; ids bit-exact vs the composite on real inputs (stable lowest-index tie-break vs `ttnn.topk`'s bitonic network; no exact ties on real inputs), weights 1.9e-7 rel vs the golden, decoder-layer and decode-trace tests green with the knob on | merged (b5d0352b988) behind `GLM53_ROUTER_FUSED=1` (default off until the 8x4 A/B) |
| `ep32l1`: EP32 L1 clash on the 8x4 | root cause: `TtEPExperts.combine_rows` kept a persistent per-layer zero constant in L1 (one per MoE layer: 24 KB per core at 32 users per row, 8 KB at 4; 42 layers add up to the 216 / 336 KB gaps of the two failures; the block never ran more than one EP layer); fix: the constant in DRAM, shared through the mesh. Block (rows 6-7, EP=8): pre-fix clash reproduced, post-fix layers 0-21 at 32 users per row 47.7 ms per step (21.0 tok/s per user, 64 users), 4 per row 31.6 ms, chain PCC 0.942 both | on `arg/glm53_ep32l1` (835a23dc959 + c0a1d410676); 8x4 e2e at 4 and 32 users per row queued |
| `mu1m`: multi-user 1 M prefill (heads -> users) | `GLM53_PREFILL_PARTITION=heads2users`: all-gather of the 16-head q over the columns, every column attends over its own slot, dummy blocks for the non-owner columns, no scratch slot; layer-3 A/B bit-exact; 16 K and 64 K at 4 users per row partitioned on the rows 0-1 block: needles found for the users, TTFT 11.8 s per 16 K user after the first | on `arg/glm53_mu1m` (ecc2ef47d27 + 5ea13946500); 1 M x 4 users per row (32 users at 1 M) on the 8x4 queued |

The fused mHC pre-block (`GLM53_MHC_PRE_BLOCK=1`) had never run on a mesh: its norm-weight read-back assumed one
device (fixed d83e20b2094, shard 0 through `ttnn.get_device_tensors`); the block A/B is queued after the 8x4 windows.

### Knee probes and the router A/B (2026-10-21 15:30-16:00)

Two more 1 M needle probes with a 160-token budget: depth 0.62 (~400 K tokens before the question) and 0.56 (~460 K)
are both found (reasoning preamble, then the key), so the 1 M miss is confined to the ~523 K-distance case while 20 K,
262 K, 400 K and 460 K distances work under the same 1 M prefix: the model-side reading stands.

Fused router top-k on the 8x4 (45 layers, bfp8, 4 users per row, `test_mesh_model -k chain`): chain PCC
0.750624 / 0.811891 / 0.827025 and the greedy ids identical to the composite's digits, traced decode step **69.8 ms**
(14.3 tok/s per user, 458 aggregate) vs 75.5 ms with the composite. `GLM53_ROUTER_FUSED=1` is the default from
4d62fa6795c (`0` keeps the composite).

### Fused mHC pre-block on a mesh: wrong and unsafe (2026-10-21 16:13)

With the weight read made mesh-safe, `GLM53_MHC_PRE_BLOCK=1` on the rows 4-5 block (45 layers bfp4, 4 users per row,
fused router) gave chain PCC 0.701 against the control's 0.824 with the greedy id wrong for every user, a faster step
(64.0 vs 79.3 ms), then a UMD exception at device close (core dump); the next job on the block timed out on the ARC
message queue and two telemetry snapshots 12 s apart showed frozen heartbeats on all 32 chips (the ARC-hang signature
of the 2026-10-20 21:41 incident). The op's kernels (`custom_ops/mhc_pre_block`) are suspected of mis-addressing the
NOC on a mesh; the knob stays off and the op must not run on a mesh before a kernel audit. Recovery per the device
rules: zero holders confirmed, one `tt-smi -r`; the EP32 8x4 e2e and the 1 M x 32-user rerun follow in
`_logs/mesh/sched_post_reset.sh` once the heartbeats advance again.

## 2026-10-21 evening: the 1 M x 32-user run, the column-3 divergence, the EP32 combine

The post-reset chain (`_logs/mesh/sched_post_reset.sh`) completed the 1 M x 4-users-per-row partitioned run
(`long_8x4_1M_4users_partitioned.log`, JSON `generated/glm53_flash_long_context/mesh8x4_N1047526.json`):

| item | result |
|---|---|
| per-user TTFT (one prompt in flight, 8 rows SP) | 494.7 / 466.6 / 465.9 / 461.7 s (2118-2269 tok/s) |
| DRAM after each prefill | 2.08 / 2.07 / 2.05 / 2.04 GiB free of 31.33 (per-user release holds) |
| traced decode step, 32 slots at 1 047 526 | 160.8 ms = 6.2 tok/s per user, 199 aggregate |
| needle (depth 0.5, gen 128) | 0/32 (the known depth-0.5 miss: the single-user d50 probe misses, d56 finds) |
| users agree | **False**: users 0-2 first token 785 'The', user 3 (column 3 of every row) token 6 "'" |
| rows agree | **False**: user 3's text differs per row (834 / 856 / 896); users 0-2 diverge across rows after ~100 chars |

Every chain is a degenerate filler continuation with a repeated number (878 for users 0-2, 896 for user 3), i.e. a
near-tie regime where any numerical difference flips tokens; but identical inputs must give bit-identical outputs, so
the user-3 / row differences are real. Exonerated since (all exact):

| check | where | result |
|---|---|---|
| DSA heads2users prefill, users 0..3, 64 K, sub-block 256, chunkings [16384] and [8192, 8192] | rows 4-5 block, `test_mesh_heads2users_prefill.py -k heads2users_long` (`GLM53_H2U_USERS=all`) | latent / pools / rings equal to the reference on every column, rows PCC >= 0.999998 |
| decode batch independence, layers 3 (DSA) and 4 (KDA + MoE), B = 4, positions 0..15, eager and traced | rows 4-5 block, `test_mesh_batch_independence.py` (new) | rows agree; changing the last or the first slot's prompt leaves the other slots bit-identical |
| full model (layers 0-3), 16 K x 4 users per row, chunked heads2users prefill, 8 decode steps | rows 4-5 block, `probe16k_2x4_4users_L0-3.log` | users 0..3 and both rows bit-identical (same 8 ids) |
| full model 45 layers, 64 K x 4 users on a block | rows 6-7 block | OOM after the build (the block holds the model at 4.1 of 4.2 GB per bank): moved to the 8x4 |

Still open: the length bisect on the 8x4 (`sched_64k_8x4.sh`: 64 K then 256 K x 4 users, 45 layers, `users agree`),
then the 1 M x 4 users rerun at needle depth 0.56 / gen 160 (`sched_1m32_d56.sh`). Candidates left: a 1 M-only
effect (memory at 29.3 of 31.3 GiB, 262144 pools per user, the decode trace captured before the prefills: "Allocating
device buffers is potentially unsafe due to the existence of an active trace") or a 45-layer-only path.

Scheduling lesson: the hand-launched 16 K probe raced the chain's zero-holder gate (a 20 s poll) and overlapped the
256-user re-measure's mesh open on rows 4-5 for its first seconds (no kernels overlapped: the probe spent 8 min on
host work; a clean rerun is queued). One scheduler owns every chip gate; new jobs are appended as gated stages.

### 256-user re-measure and the EP32 combine analysis

Hybrid dense pass, 45 layers, fast mHC + fused router (today's defaults): traced decode step **115.0 ms** at 256
users (8.7 tok/s per user, 2226 aggregate; 121.6 ms on 2026-10-20), greedy id == host head for every user.

Agent analysis of the layer-4 tracy profiles (`_agents/ep32_combine/ANALYSIS.md`, `per_chip_a2a.csv`): the EP32
`all_to_all_combine` runs on 2 cores (= the 2 routing planes of axis 0) scanning every gathered token with one NOC
read + barrier per row and one fabric packet per 8 KB row: 338-892 us per chip at 32 users/row for ~0.47 MB moved,
2-5 % of the 2 x 24.2 GB/s link rate; its spread is the column's expert-stage skew (it ends when all 8 senders have
delivered). The e2e EP32 figure (135.8 ms) was a hot-spot artefact: `test_mesh_model.py` gave every user the same
token, so all 256 users shared one top-8 and 1-2 chips per column returned 256 rows each (~1.0-1.3 ms/layer vs
0.34-0.42 with diverse routing, +25-38 ms over 42 MoE layers); the hybrid dense pass is routing-independent. In-trace
op gaps are 64 us/layer for both layouts (nothing host-side).

Taken: (1) `GLM53_EP_COMBINE=a2a` (default `op`): the zero-padded expert output [1, 1, rows*T, H] exchanged densely
along the rows with `all_to_all_async_generic` + `fast_reduce_nc` over the senders, dropping the slot clone /
untilize / tilize; expected 338-892 -> ~100-130 us per layer (T % 16 == 0 only, so 4 users/row stays on the op);
(2) the e2e perf loop feeds one random token per user (`GLM53_MODEL_PERF_IDS=distinct`, default). Validation and
the distinct-id re-measures of both layouts at 32 and 256 users are the chain's last stage (`sched_ep_combine.sh`);
the README decode rows get refreshed from them.

**64 K x 4 users per row, 45 layers, blocks prefill path (19:13, `long_8x4_N65536_4users_partitioned.log`):** needles
32/32, rows agree, users agree (all 32 chains identical: 'amber-falcon-7391'); per-user TTFT 155 s (420 tok/s: the
default `GLM53_PREFILL_EXPERTS=blocks` path, 5x slower than the regions path the 1 M run used at 2.2 K tok/s), decode
90.0 ms = 11.1 tok/s per user (356 aggregate) on the stream expert op. So the column-3 divergence is absent at 64 K
with the full model; the bisect continues on the regions path (`sched_bisect_v2.sh`: 64 K, 256 K), which the first
bisect scripts had omitted.

**64 K x 4 users per row, regions prefill path (19:23, `long_8x4_N65536_4users_regions.log`):** TTFT 32.5 / 27.4 / 27.3 /
27.3 s (2016-2402 tok/s), needles 32/32, rows agree, users agree; decode 148.2 ms = 6.7 tok/s per user through the
regions expert kernel (no stacked copy; 13.2 GiB free after the prefills, so `GLM53_REGIONS_KEEP_STACKED=1` fits and
gives the 90 ms block-op decode). Bisect continues at 256 K. To-do once the chips are free: make `regions` the
default prefill path with the stacked copy kept whenever the memory planner says it fits (validate the MoE unit test
and the traced prefill on a block first: the regions path must be trace-safe for prompts of 33..2048 tokens).

**256 K x 4 users per row, regions path (19:40, `long_8x4_N262144_4users_regions.log`):** TTFT 135.8 / 109.9 / 109.8 /
109.0 s (1928-2403 tok/s), needles 32/32, **users agree, rows agree False**: every chain answers
'amber-falcon-7391</think>' and then rows 2-3 (users 1-3) / row 5 (user 0) pick token 154842 where row 0 picks 154827
(a special-token near-tie). So rows are not bit-identical at long context even with every slot correct; the users of
a row are. Hypothesis: the hybrid MoE sums the expert partials over the 8 rows in a ring (reduce-scatter), so each
row accumulates its tokens' sums in a different fp order (a 2-row block hides this: a+b == b+a); near-tie steps then
flip per row. The 1 M user-3 difference needs a within-row asymmetry on top. Two knobs added to the long test for the
next 1 M experiments: `GLM53_LONG_USER_ORDER=3,0,1,2` (is it column 3 or the last prefill / the memory state?) and
`GLM53_LONG_REPEAT_USER0=1` (prefill user 0 again at the end: same first token and last-chunk logits = the 1 M prefill
is deterministic in that memory state).

**1 M x 4 users per row, needle depth 0.56, gen 160, regions path (20:20, `long_8x4_1M_4users_d56_g160.log`):**
TTFT 489.1 / 460.7 / 460.4 / 456.4 s (2142-2296 tok/s), **needles 32/32, rows agree, users agree** (32 identical
160-token chains), decode 152.1 ms = 6.6 tok/s per user (210 aggregate) at 1 047 612 — the README's 1 M x 32 row.
The column-3 first-token difference of the depth-0.5 run did not reproduce on this prompt: it is prompt-specific
(the degenerate depth-0.5 continuation), i.e. a near-tie flip from a small within-row asymmetry, not a broken slot.
Next: the same depth-0.5 prompt with the users prefilled in the order 3,0,1,2 plus a repeat of user 0 at the end
(`sched_1m_order_d50.sh`): same flip on user 3 = a column-3 property; flip on the user prefilled last = order / memory
state; a repeat of user 0 with a different first token or logits = non-deterministic prefill at 1 M.

**1 M x 4 users, depth-0.5 prompt, prefill order 3,0,1,2 + repeat of user 0 (21:09,
`long_8x4_1M_4users_order3012_repeat0_d50.log`):** all four first tokens 785 'The' (user 3 prefilled first, user 2
last: the original run's user-3 "'" did not follow the column or the last prefill); REPEAT user 0 after four 1 M
prefills: same first token, last-chunk logits max abs diff 0 (the 1 M prefill is bit-deterministic in that memory
state). Decode (64 tokens, degenerate depth-0.5 continuation): in EVERY row users 0, 1, 2 are bit-identical and user
3 diverges (at token 5 / 48 / 5 / 27 / 21 / 5 / 21 / 21 per row). So slot 3 (column 3) is treated differently in the
DECODE step by a last-ulp amount that only shows at near-ties; the d56 prompt (160 identical tokens for all 32) and
256 K (8 tokens) never hit one. Candidates: the batched index scorer / two-stage segmented top-k (score ties at the
top-512 boundary broken differently for the last user's segments), or the heads->users all-to-all's last chunk.
Next: a single-chip tie-break test of the scorer + top-k with 4 identical users (tests/custom_ops), and a 1 M-step
dump of user 0 vs user 3's selected pool ids.

**Slot-3 ledger, 21:26-21:46 (blocks):** same prompt in every slot -> bit-identical outputs for layers 3 and 4, eager
and traced, positions 0..15 (`indep_2x4_sameprompt_*.log`); partitioned sparse decode of layer 3 with the R4097
prompt in all 4 users (heads2users), 12 steps incl. trace replay -> slots byte-equal, worst PCC vs HF 0.999953
(`sparse_partitioned_same_user_h2u.log`, `GLM53_SPARSE_SAME_USER=1`); full model 45 layers bfp4 at 16 K x 4
users with a filler-only prompt (`GLM53_LONG_NO_QUESTION=1`) -> the model answers confidently ("I need the actual
question ..."), all slots and rows identical over 64 tokens. Status: the slot-3 divergence is observed only at 1 M
on the depth-0.5 prompt (a degenerate near-tie continuation); the 1 M prefill is bit-deterministic; no layer-level
or <= 256 K check reproduces it. Parked as low impact (degenerate regime only); next probe if needed: a 1 M step dump
of user 0 vs user 3's selected pool ids and attention output per layer.

**128-token first token vs HF on the 8x4 (accuracy regression, open):** every 8x4 TTFT check since 17:35 mismatches
the host-head argmax (EP32 builds: token 39030, the same as the 10-20 evening runs; hybrid dense B=32 at 21:26:
99484, chain PCC 0.895-0.917 vs 0.943 this morning), while the hybrid sparse B=4 bfp8 build matched at 07:44-09:38
and the block (bfp4, B=4) matches with the fused router on AND off (bit-identical chain PCC 0.823420:
`ttft_2x4_router_fused_{0,1}.log`). So not the router. Queued on the 8x4 before the perf re-measures
(`sched_ttft_ab.sh`): hybrid dense B=32 with the fused router 0 / 1 and hybrid sparse B=4 on today's code.

**128-token first token: root cause and fix (22:28).** 8x4 A/B (`ttft_8x4_*.log`, 45 layers bfp8, composite = the
fused router off):

| build | router in the prefill slab | first token | chain PCC vs HF rows |
|---|---|---|---|
| hybrid dense, 32 users/row | composite | 102647 = HF | 0.895744 (= v17dense this morning) |
| hybrid dense, 32 users/row | fused op | 99484 | 0.894983 |
| hybrid sparse, 4 users/row, fix applied | composite | 102647 = HF | 0.900901 (= fresh_fast_v19 this morning) |

A 128-token prompt sequence-parallel over the 8 rows gives every chip a 16-row slab, which satisfied the fused
router's "decode shape" condition (T <= 32) although the op was only ever validated against the composite on decode
inputs; its weights differ in the last bf16 ulp and the first token sat at a near-tie. The 2-row block (64-row
slabs) never took the fused path, hence the bit-identical block A/B. Fix 6e5d736cd14: `TtMoE.forward(prefill=True)`
from the decoder layer's prefill wrap keeps the composite router on prompt slabs; decode keeps the fused op (its
ids are bit-exact there). The EP32 builds' wrong first tokens (39030 / 3147, PCC 0.79-0.82) predate the fix; the
EP e2e re-measures that follow run with it and report whether EP32's own prefill path is also off.

### Optimization phase kick-off (22:30-22:50): decode ledger, MTP head, first two items

Agent deliveries: `_agents/decode_ledger/LEDGER.md` + `ops_per_layer.csv` (per-op decode ledger of a KDA+MoE and a
DSA+MoE layer from the tracy profiles and the code: 65 / 108 ops per layer, already at the DeepSeek-V3 Galaxy
reference's op count, so the gap to 0.67 / 1.17 ms per layer is per-op time: single-core mHC pre-block ops, the
2-core combine, 8-row serialized collectives, 64 us/layer of dispatch gaps from generic ops built with one
CoreRange per core); `_agents/mtp_head/PLAN.md` + `mtp_reference.py` (layer 45 = a DSA layer with its own MoE and
indexer, enorm/hnorm/eh_proj glue, plain residuals; CPU reference verified against HF's own layer-45 forward:
block PCC 0.99979, top-1 127/128; draft levels appended to the traced decode graph, verify as one B*(K+1)-row step,
KDA state recomputed from cached projections on accept; gain model 1.5-2x at <= 32 users, off at 256).

Implemented (3cfcdc399f5, validation queued at the chain's end, `sched_ledger_validate.sh`):

| item | change | expected |
|---|---|---|
| B6 rectangular core ranges | `custom_ops/core_ranges.rect_core_ranges` (CPU-tested) in `kda_decode` and `paged_pool_gather` | 34 + 15 us of dispatch gap per layer -> < 1 us: ~1.3 ms per step at 32 users |
| C1 EP combine as reduce-scatter | `GLM53_EP_COMBINE=rs`: `reduce_scatter_rows(out, dim=2)` on the zero-padded expert output (T % 32 == 0) | 338-892 us combine -> ~130-144 us per MoE layer: -13..-22 ms at 256 users |

Measured tonight for the record: EP32 with the a2a combine at 256 users, distinct ids: 126.3 ms per step (7.9
tok/s per user) and traced TTFT 149 ms (the a2a branch is taken; the 8-chip generic all-to-all is itself
latency-serialized, ~1.1 ms per MoE layer, hence not the predicted 90-100 ms). Next in the ledger's order: A3
(sharded multicore mHC norm, -3.4 ms), B7 (TP all-reduce as all-gather + fast_reduce_nc, -1.5 ms), B5 (DSA fused
state + folds, 50 -> 14 attention ops), then A1 (fused mHC pre-block) after its NOC audit.

**One-run re-measures with distinct tokens (22:47-23:05).** EP32 op combine, 256 users: 139.2 ms (7.2 tok/s per
user), traced TTFT 171 ms, 128-token first token still wrong (39030, PCC 0.788: EP32's own prefill path, open).
Hybrid, 32 users (4/row): **83.1 ms** per step = 12.0 tok/s per user (385 aggregate) against 69.8 ms with the same
token on every slot (the stream expert op was flattered by one shared top-8); traced TTFT 266 ms at 128 tokens;
first token == HF (0.900901) with the router fix. README rows updated. MTP step 1b (`_agents/mtp_head/STATUS.md`):
level-1 draft acceptance 0.55 on an off-policy paragraph, 0.91 on an answer-like region, 0.66 over 393 rows, half of
the misses main-head near-ties (gap < 1 logit); TT block `tt/mtp.py` + `tests/test_mtp.py` committed untested
(03990ef08f4), first chip-0 run queued (`sched_mtp_test.sh`).

**Envelope point 1: 8 users per row x 512 K (00:03, `long_8x4_N524288_B8_envelope.log`):** first run with 2 slots per
chip in the heads2users layout; 64-slot model built in 602 s, per-prompt TTFT 240.2 / 225.0 / 225.6 / 225.0 / 225.3
/ 225.0 / 223.1 / 223.1 s (2187-2354 tok/s), 1.95 GiB free after the 8 prefills (planner: fits), decode 178.2 ms =
5.6 tok/s per user (64 slots, 359 aggregate). Needles 27/64 with gen 32: the chains are reasoning text ("We need
answer passphrase only ... There is sentence: \"The secret passphrase f" / "The user is asking for the secret
passphrase ... Looking through the text, I ca") cut before the answer, i.e. the GLM53_LONG_GEN >= 128 rule at >= 512 K
was not applied in the envelope scripts; openers split 'The' (users 0, 1, 7) / 'We' (users 2-6) across slots and rows
(a near-tie, not per column or per order). Gen-160 rerun queued after the MTP test (`sched_envelope512_g160.sh`).
EP32 at 32 users with distinct ids: 71.2 ms (14.1 tok/s per user), traced TTFT 172 ms, first token 39030 (EP32
prefill, open).

**Logits-level accuracy on natural text (00:35-00:52, `logits_8x4_nat_mhc_{fast,wrap}.log`, dump L45_NAT_S128, 45
layers bfp8, 8x4):**

| mHC | logits PCC mean / min | top-1 | top-5 overlap | KL mean (nats) | first token |
|---|---|---|---|---|---|
| fast (default) | 0.9938 / 0.902 (pos 54) | 89.8 % (115/128) | 94.1 % | 0.040 | 11505 = HF |
| composite wrap | 0.9942 / 0.910 (pos 54) | 90.6 % (116/128) | 94.1 % | - | 11505 = HF |

CPU floors (`tests/test_logits_check.py`): bf16 hidden / bfp8 head vs fp32 = PCC 0.99998, top-1 128/128; HF's own
bf16 head flips 2/128 top-1s (98.4 %). The device's misses are HF top-2 swaps at near-ties (device top-1 always in
HF's top-5); the prefill is run-to-run deterministic (max |dlogits| 0). The fast mHC costs 0.0003 of logits PCC and
one near-tie position: kept. The residual-stream PCC (0.943 composite vs 0.901 fast at 128 tokens) was the
depth-chaos amplification of that. Test bars set to the measured floor (logits PCC >= 0.98, top-1 >= 0.85).

**Envelope point 2: 16 users per row x 256 K (01:33, `long_8x4_N262144_B16_envelope.log`):** 4 slots per chip, 128-slot
model; per-prompt TTFT 123.1 then 109-111 s (2126-2400 tok/s), 1.63 GiB free after the 16 prefills per row, decode
188.1 ms = 5.3 tok/s per user (128 slots, 680 aggregate); **needles 128/128, users agree** (rows differ at a
near-tie as usual). README row added.

**Envelope point 3: 32 users per row x 128 K (02:09, `long_8x4_N131072_B32_envelope.log`):** 8 slots per chip, 256-slot
model; per-prompt TTFT 63.0 then 54.6-55.2 s (2078-2395 tok/s), 1.12 GiB free after the 32 prefills per row, decode
199.0 ms = 5.0 tok/s per user (256 slots, 1287 aggregate); **needles 256/256, rows agree, users agree**. The planner's
users x ISL envelope is now measured end to end at 32 x 1 M, 64 x 512 K, 128 x 256 K, 256 x 128 K (free DRAM 2.04 /
1.95 / 1.63 / 1.12 GiB). Decode per user at the envelope: 6.6 / 5.6 / 5.3 / 5.0 tok/s: the step grows with the slot
count (160.8 -> 178.2 -> 188.1 -> 199.0 ms) far less than the batch, i.e. the long-context decode is latency-bound
(the ledger's per-op items), and the aggregate scales 210 -> 359 -> 681 -> 1287 tok/s.

**README 64 K x 4 users per row, one config (02:25, `long_8x4_N65536_4users_readme.log`, regions prefill +
`GLM53_REGIONS_KEEP_STACKED=1`):** TTFT 33.3 / 27.3 / 27.3 / 27.2 s (1964-2400 tok/s), decode 91.4 ms = 10.9 tok/s
per user (350 aggregate) on the block expert op, needles 32/32, rows and users agree, 3.83 GiB free after the
prefills. This is the configuration `GLM53_PREFILL_EXPERTS=auto` selects at 64 K (block validation next).

**Vision demo, today's defaults (02:28, `mm_2x4_demo_defaults.log`):** ISL 286 (256 image tokens), TTFT 1147 ms
(tower 60 ms), decode 73.7 ms = 13.6 tok/s per user (27 aggregate over 2 slots), first token == HF. README row.
**`GLM53_PREFILL_EXPERTS=auto` on the rows 6-7 block (02:20-02:38, `regions_auto_2x4_e2e.log`):** the planner drops
the stacked copy (45 layers bfp4 do not fit twice on 8 chips, as documented) -> regions prefill + regions decode; the
traced prefill works through the regions path (128 tokens: traced 345 ms, eager 705 ms), first token 102647 == HF
(chain PCC 0.8738), chain decode 133.5 ms at 4 users/row; the chain test only failed its 0.99 residual-PCC bar
because the scheduler did not lower it (every other e2e run uses 0.5). The 8x4 comparison of auto (regions + stacked
copy) against the blocks default on the 32-user short-prompt row is queued (`sched_auto_8x4.sh`); the default
flips if TTFT and decode are at least equal and the first token still matches.

**Ledger validation (03:21-03:40):** rectangular core ranges: 25 kda_decode / paged_pool_gather / core_ranges unit
tests pass on chip 0 (`ledger_unit_chip0.log`); reduce-scatter EP combine (`GLM53_EP_COMBINE=rs`): block EP MoE vs HF
0.999923 (= the op path, `ep_combine_rs_moe_vs_hf.log`); 8x4 EP32 e2e at 256 users with distinct tokens: **123.4 ms**
per step (8.1 tok/s per user, 2075 aggregate; op combine 139.2, a2a 126.3; hybrid dense 115.5), in the ledger's
predicted range. `rs` is now the EP combine default (T % 32 == 0; 4 users/row stays on the op until the down writer's
row stride remap). The hybrid 32-user run with the rectangular ranges (B6) follows.

**B6 result (03:55, `e2e_hybrid_B4_rect.log`):** hybrid 32 users with rectangular core ranges in `kda_decode` and
`paged_pool_gather`: 83.08 ms per step vs 83.07 before: no measurable change. The ledger's 64 us/layer "dispatch
gap before the generic ops" is therefore not the one-range-per-core program descriptor (or trace replay already
amortises it); the change is kept (unit tests pass, simpler descriptors) but the 1.3 ms claim is withdrawn. To
locate the gap: a layer-4 tracy profile of the traced step with the change, comparing the op start times around
`kda_decode` and `paged_pool_gather` with `report_prof_ep32_ep_b4`.

**MTP block, first device run (03:58, chip 0, `mtp_test_chip0.log`, `tests/test_mtp.py`):** built in 58 s; prefill 16
rows vs the CPU reference: glue PCC 0.999914, block output 0.998332, logits 0.998476, top-1 16/16, device argmax ==
host argmax 16/16; decode step at position 16: block 0.999837, logits 0.999653, greedy == reference on both slots,
slots bit-equal. Steps 1-2 of `_agents/mtp_head/PLAN.md` are done; next: the level-1 draft inside the traced decode
graph of the mesh model (step 3), then the batched verify step (4) and the KDA state commit (5). Code pushed
(upstream 380b1b7da30) together with the rs combine default, the core-range helper, the auto prefill mode and the
logits-level accuracy tooling.

**Envelope point 1 rerun with gen 160 (04:40, `long_8x4_N524288_B8_envelope_g160.log`):** needles **64/64**; TTFT
240.1 / 224.9 / 224.7 / 224.8 / 225.1 / 224.9 / 222.8 / 222.8 s, decode 184.6 ms = 5.4 tok/s per user (64 slots,
347 aggregate). The openers split 'We' (users 0-3) / 'The' (users 4-7) this time against 'The' (0, 1, 7) / 'We' (2-6)
in the first run: the first token at this prompt is a near-tie that flips across runs and slots while every chain
reaches the passphrase; within a run the prefill is bit-deterministic (the repeat-user-0 check), so the cross-run
flip is the near-tie plus whatever differs between processes at the last ulp (allocation order, stale padding).
README row updated. The users x ISL envelope is now needle-verified at all four points.

**auto vs blocks on the 8x4, 32 users, short prompt (04:56, `e2e_hybrid_B4_auto.log`):** planner keeps the stacked
copy; traced TTFT 266 / 280 ms (= blocks 266), eager 976-1004 ms (blocks 1383), residual chain PCC 0.949673 (blocks
0.900901; the regions kernel is closer to HF on the residual), first token == HF, decode **86.6 ms** vs 83.1 with
the blocks default (the second expert copy). Decision (93b9d6b233a): `GLM53_PREFILL_EXPERTS=auto` is the default,
resolved by the mesh model as blocks up to 4 K positions and regions (+ stacked copy per the planner) above, so every
README row's configuration is the default one; a bare layer treats auto as blocks. The chain of 2026-10-21/22 is
complete: AUTO 8X4 DONE.

## 2026-10-26 device validation of the device-free work (chain `sched_validate_devicefree_work.sh`, 29 stages, 15:08-17:52)

One run per stage; traced numbers; every e2e reports TTFT and decode from the same run; first token == HF unless noted.

| stage | result |
|---|---|
| mHC A3 chip 0 | sharded norm vs single: wrap output PCC 0.999998 (T = 4 and 32), traced replay == eager; traced wrap 246 -> 210 us (sharded), 206 us (sharded_cast); the A4 batched reference graph differs from the fused Sinkhorn by 5e-4 (SFPU approximations; bar widened to 1e-2) |
| mHC units chip 0 | 3 pre-existing failures: `build_mhc(impl="fast")` returned the split pair since eeb7010f222; fixed (explicit impl = that class); decode-trace tests pass |
| C1 rs remap | chip-0 row chunks 3/3, block remap vs op 2/2 |
| MTP block chip 0 | 1/1 after the steps 3-5 edits |
| DSA state op chip 0 | 9/9; the chip-0 A/B of the wiring failed in its setup: the single-chip full-width q_b stem overflowed L1 (2.3 MB of CBs) -> `prefill_linear` picks an L1-aware row block (TP4 widths unchanged) |
| pre-block watcher chip 27 | unit 8/8 and one decoder layer 5/5 under the NOC sanitizer: no violation (hang exclusion) |
| pre-block A/B chip 27 + block | 12/12 and 4/4: hn PCC 0.999996, post / comb lanes bit-identical, two instances, trace replay, bit-identical on all 8 block chips |
| B7 gather all-reduce block | 7/7 |
| prefill selection block | `untilize_with_unpadding` rejected the ROW_MAJOR scorer output -> every sparse prefill on the branch was broken since c1e1b46f679; fixed (tile-major only); re-run queued |
| prefix arena block | same failure; re-run queued |
| decoder block, both knobs | 3/3 vs HF |
| block e2e, 4 users/row | decode 78.4 ms with the knobs vs 80.7 ms without; TTFT part overflowed the 256 MB trace region (the block needs 512 MB with the traced prefill ladder); re-runs queued |
| MTP spec decode block | step 3 (traced draft == eager) passed; the verify test hung: the plain decode trace was captured before the spec graphs compiled their 8-row collectives (late-compile trap, second instance) -> host spin, SIGTERM, FW init failure, one `tt-smi -r`; tests now warm up before any capture; re-run queued |
| B5 block A/B (partitioned) | output PCC 0.99953 (bar 0.9998): rings, selected ids (2 near-tie flips) and latent agree, but one whole 32-row pool block differs per sparse user (dense user fine); row-level diagnostic + `latent` fallback queued |
| batch independence, B5 knob | 2/2 |
| 22-layer EP chain, pre-block on / off | 33.3 vs 37.7 ms decode, identical first token and prefill PCC |
| block logits gate, pre-block | OOM at model build (the 45-layer bfp4 block has no headroom for the 512 MB trace region) -> gate moved to the 8x4 (queued) |
| 8x4 e2e, 32 users, knobs | decode 80.55 ms = 12.4 tok/s/u, 397 agg (ref 83.1-86.6); TTFT 314 ms vs 266: the 16-row prefill slabs passed the gather all-reduce's <= 32-row gate and the mHC split's one-tile test -> `GLMMesh.prefill_scope()`; validation queued |
| 8x4 e2e EP, rs remap | decode 67.68 ms = 14.8 tok/s/u, 473 agg (ref 71.2); TTFT 152 ms (unchanged); first token != HF (the known EP prefill gap) |
| 8x4 e2e, 256 users, knobs | SIGKILLed at layer 37 by the host OOM killer (594 GB box, 134 GB hugepages, ~390 GB page cache, 3.6 GB free); re-run queued |
| 8x4 logits gate, knobs | PCC 0.9948, top-1 95.3 %, top-5 93.4 % (ref 0.9942 / 90.6 / 94.1) |
| 8x4 64 K needle, knobs | 32/32, rows and users agree; TTFT 25.4 s per prompt (ref 27.3), 2580 tok/s prefill, decode 92.1 ms = 10.9 tok/s/u |
| 8x4 64 K needle, DSA state knob | 32/32; decode 90.9 ms = 11.0 tok/s/u (-1.2 ms, the ledger's 11 x 145 us) |

Defaults flipped: `GLM53_MHC_NORM=sharded`. Pending their re-runs: `GLM53_TP_ALLREDUCE=gather` (TTFT with the prefill
scope), `GLM53_PREFILL_ABLOCK=2048` (selection bit-identity), `GLM53_DSA_DECODE_STATE=1` (the pool-block diagnostic),
`GLM53_MHC_PRE_BLOCK=1` (the 8x4 logits gate). Follow-up runner: `sched_validate_followups.sh` + `followups.queue`.

### 2026-10-26 evening: roofline program, follow-up results, the wedge

Follow-up runs (frozen snapshots once the live tree was under agent edits): knob-off 8x4 twin 79.75 ms / TTFT 297 ms (sharded
norm default); gather all-reduce + prefill scope 80.57 / 298 (the gather gives no decode gain at 4 users/row: stays off); single
norm 83.06 / 266 = the 10-22 reference bit-for-bit, so the sharded norm = -3.3 ms decode and +31 ms of 128-token TTFT (16-row
prefill slabs take the fast mHC path) -> decode-only norm selection (mHC agent, committed). 64 K needle with the DSA state
knob: 32/32, 90.9 vs 92.1 ms. Roofline model (`roofline.md`, `tt/roofline.py`): current-graph floor 32.3 ms at 32 users (31
tok/s/u) vs 80.6 measured; 25.2 ms with the fused pre-block, B5 and bfp4 gate|up; ranked actions E0 stream experts in decode
-13.9, A1 -10.4, H2 -4.7, E1 -4.2, A4 -4.0, H1 -2.9, C12 -2.9, B5 -1.6, B8 -1.5, P1 -1.3, K3 -1.2. Agents (device-free):
KDA v2 (`GLM53_KDA_DECODE_V2`, est. -0.25 / -1.4 ms at 32 / 256 users; the fused KDA path was already 7 ops with qkv at the
DRAM roofline), mHC (`GLM53_MHC_PRE_SINKHORN=fused`, `GLM53_MHC_{PRE,POST}_SPLIT=extras`, `GLM53_MHC_KSUM=fast`; D13 not a
win after A1), MoE (in progress). Incidents: a live-tree B=32 run compiled a half-edited KDA kernel (ct arg 11 of 11) ->
device stages now run from frozen worktrees; the batch-1 run hung after its model build (19:10; live tree or the sharded
norm at T=1, both untested) and the second `tt-smi -r` of the day FAILED (warm_reset.cpp:337, PCIe AER non-fatal errors,
"Query mappings failed on device 0: No such device") -> host recovery needed; queue preserved (30 entries, T=1 mHC A/B
first), restart recipe in the memory note.

### 2026-10-26 night: after the reboot, the stream-expert flip, the roofline program

Post-reboot (user host reboot ~19:45): 32 PCI devices, (8,4) probe 6.4 s; every device stage runs from a frozen snapshot
(wt_frozen..wt_frozen6). mhc A/B at T = 1, 2, 4, 32: 4/4 (the batch-1 hang was the live-tree compile). Measured on the 8x4
at 4 users/row, 45 layers, 128-token prompt (one run each, first token == HF): default (sharded norm + prefill-scope single
norm) 79.59 ms / TTFT 270 ms (prefill bit-identical to the 10-22 reference); + GLM53_STREAM_EXPERTS=1: 69.1 ms = 14.5
tok/s/u, 463 agg / TTFT 135 ms; logits gate with the stream op PCC 0.9946, top-1 93.0 %, top-5 93.1 %; 64 K needle 32/32,
decode 92.1 -> 80.3 ms, BUT prefill 25.4 -> 139 s (the stream flag forced the blocks prefill path) -> fixed (regions keeps
its lists from a transient block-layout load) and the stream op is now the DEFAULT for decode-shaped rows (unset = not
GLM53_DENSE_EXPERTS). README 32-user row booked: 135 ms / 14.5 / 463. Roofline (`roofline.md`, `tt/roofline.py`): current
graph floor 32.3 ms at 32 users (31 tok/s/u) vs 69.1 now; plain after the ranked actions ~35 ms (28.5); spec-decode mode:
the v1 loop is break-even at K=3 (commit blends 23.4 ms) -> SP1 gated commit committed (1.31x at K=3, 1.63x at K=1
modelled), SP2 fused T=K+1 kernel in progress (C ~3.4 ms). Agents committed (knobs off unless stated): KDA v2
(GLM53_KDA_DECODE_V2), mHC fused Sinkhorn / splits / K-sum (GLM53_MHC_PRE_SINKHORN=fused, GLM53_MHC_{PRE,POST}_SPLIT=extras,
GLM53_MHC_KSUM=fast), MoE C10/C11/C12 + weight-read plan + EP placement, hybrid glue H1/H2 (GLM53_HYBRID_GATHER=compact,
GLM53_HYBRID_COMBINE=allreduce), B8a (GLM53_POOL_GATHER=dense_skip), C2 experiment; E1 expert-op fixed cost + dispatch form
for 32 users/row in progress. Device queue (43 stages, frozen snapshots): defaults-only 64 K / 32-user / batch-1 runs first
(README rows), then B32, B5 diagnostics, re-runs, spec blend/gated, pre-block gates, all-knobs gates, KDA / mHC / MoE /
glue A/Bs with their logits gates.

## 2026-10-22 device-free deliveries (machine in use elsewhere; nothing run on chips)

| track | delivered | default | device validation (`_logs/mesh/sched_validate_devicefree_work.sh`, starts only on `touch GO_VALIDATE`) |
|---|---|---|---|
| A3 mHC norm on 32 cores | `GLM53_MHC_NORM=sharded|sharded_cast|single` in the exact decode collapse (+2 ops, expected -70 us/layer = -3.2 ms/step); host reference test 19/19 | `sharded` since 2026-10-26 (chip-0 A/B PCC 0.999998, traced wrap 246 -> 210 us; block decoder + e2e; 8x4 e2e 80.6 ms at 32 users; logits 0.9948 / top-1 95.3 %; 64 K needle 32/32) | passed 2026-10-26 |
| A1 fused mHC pre-block audit | `mhc_pre_block_audit.md`: NOC / dataflow audit of the fused pre-block behind the 2026-10-21 mesh PCC 0.70 + ARC hang. Verdict: the 0.70 is NOT an addressing / race hazard (deterministic and chip-identical on all 8 users, the block's 12x10 grids have no idle cores, the step time matched the op's 102 us): either depth-chaotic rounding through the 45-layer bfp4 chain on the random prompt or a model-level defect the unit test cannot see (it never compares with the composite path); the mesh test decides. The ARC hang is not attributable to the kernels (no path can target a non-worker core or an address outside L1; the same board-wide freeze happened 2026-10-20 without any custom op). Fixed anyway: H1 idle-core gather race on grids with more cores than column tiles (13x10: an idle core's unordered round-2 zero tile counts toward the leader's round-1 wait -> stale slots on every launch, a lost increment -> deadlock; reproduced by a discrete-event model of the three kernels, fixed by waiting for the pre tile), H2 static pack cache reset per launch, H3 `golden()` to_torch on a mesh, H4 device-bound tables asserted in `__call__`; op.py arg packing factored into device-free `plan_work` / `cb_plan` (same vectors, same program hash); 33 host tests; the patched kernels syntax-compiled with the JIT toolchain | `0` | watcher NOC sanitizer on one chip without fabric (hang exclusion), `test_mhc_pre_block_mesh.py` on chip 27 then all 8 block chips bit-level vs the composite through the model's `_pre` (+ two instances, trace replay), 22-layer EP block chain PB=1 vs PB=0 and the block logits gate, then the 45-layer chain and the 8x4 (`_agents/mhc_a1_audit/AUDIT.md`) |
| A4 Sinkhorn | finding: already ONE fused program per token tile (all 32 users); `GLM53_MHC_SINKHORN=batched` is a reference graph (+250-450 us), the real -94 us/layer needs a kernel change | `kernel` | none |
| B7 TP all-reduce | `GLM53_TP_ALLREDUCE=gather` = all_gather_cols + fast_reduce_nc (expected -1.5 ms/step) | `ccl` | `test_ccl_allreduce_gather.py` on the block, then e2e |
| C1 for 4 users/row | `GLM53_EP_RS_REMAP=1`: down-writer row-stride remap so the rs combine serves T < 32 (expected -2..-14 ms/step at 32 users under EP32) | `0` | `test_expert_stream_rs_t4.py` (chip 0 row chunks, block remap vs op), EP e2e |
| prefill selection | one attention block per `GLM53_PREFILL_ABLOCK` rows (2048 batches a slab), untilize_with_unpadding, cached bounds: ~1-3 % of a 16 K chunk; `GLM53_PREFIX_INCREMENTAL=1` arena (11 x ISL KiB resident: 0.69 GiB at 64 K, 11 GiB at 1 M) | ABLOCK = QBLOCK (old form), incremental 0 | `test_prefill_select_batched.py`, `test_prefix_incremental.py` on the block, 64 K needle on the 8x4 |
| vLLM generator | `tt/generator_vllm.py` + `tt/vllm_config.py` (plugin contract, logical->physical slot indirection, column-aware placement, logical block pool over device-managed pages, traced/chunked prefill planner), 17 host tests; `max_num_seqs` = users per row until per-row prefill ownership | n/a | plugin registration + (2,4) reduced smoke, then 8x4 (`_agents/vllm_glm53/STATUS.md`) |
| PP study | `pp_design.md`: no decode gain at equal batch; capacity + prefill-ceiling layout for later | n/a | per-row 6-layer prefill rate measurement |
| vLLM per-row prefill ownership | design study `row_ownership.md`: recommended = waves of row-local chunks (one chunk per row from different prompts: row-local KDA scan, the single-row DSA prefill, expert collectives unchanged since they are per token, idle rows padded into dummy targets) with SP kept for lone prompts (a wave makes a lone prompt's TTFT 3-8x worse); aggregate unchanged at W = 2 K rows per row (2.2-2.4 K tok/s), 2.6-6.5 K at W = 8 K, 2.9-7.3 K at 16 K pending the one-row chunk time vs W (the decisive measurement); envelope unchanged at W = 2 K (cells become distinct requests), +2 K of W = ~0.5 GiB/chip; critical path = the ttnn KDA ops need `sequence_parallel_size=1` on a multi-row mesh (7 files); phase 1 = uniform slot index per wave, phase 2 = per-chip page-table views for waves at full occupancy. Host slice: `ServingGeometry.wave_rows / wave_affinity / sp_fallback`, wave-affine `SlotTable.allocate`, `WavePlanner`, generator wave path behind `GLM53_VLLM_ROWS_INDEPENDENT=1` (refuses a model without `prefill_wave`); 22 host tests | off (SP path) | model-side `prefill_wave` / `prefill_traced_wave` not written; block tests (same prompt SP vs wave, two prompts per wave bit-identical per row, idle-row mask, traced wave), then 8x4 (first token vs HF per row, 64 K needle with 8 needles per wave, 32 distinct prompts at 1 M) |
| MTP steps 3-5 (+ the host loop of 6) | `tt/spec_decode.py`: `SpecDecoder` with three traced graphs (V = verify the B(K+1) candidate rows + MTP level 1 on every row, C = KDA state + pooled-index commit, D = MTP levels 2..K from the selected row by a one-hot gather), host greedy accept, K policy `spec_draft_len` (B(K+1) <= 32: 1->31, 4->7, 8->3, 16->1, 32->0 = plain), `prefill_with_mtp`; `tt/mesh_model.py` draft step (`decode_step_with_draft`, `capture_with_draft`); `tt/mtp.py` `block_rows`, multi-row `prefill_rows`, `prefill_cache`, `ids_row`; 15 host tests pass. Departures from the plan: V and D never write rings or pools (a batched ring write of a rejected row would clobber a committed slot before its pool completes), C replays `maintain_decode` per candidate in order with rejected rows routed to dead ring rows 4..31; KDA commit = replay the K+1 launches on an S_0 backup and blend with 1.0/0.0 masks (exact; one commit per verify; ~141 MB backup per chip at 4 users/row). Dense range only (p + K <= 2050), non-partitioned caches, greedy only | off (nothing calls it) | `test_mtp_spec.py` on the block (`GLM53_MESH_SHAPE=2,4 GLM53_MODEL_LAYERS=0-7 GLM53_MODEL_B=2 GLM53_SPEC_K=3`): step-3 trace drafts == eager MTP, verify / commit vs the sequential plain path (KDA state PCC > 0.9999, ring rows), 64-token greedy == plain up to near-ties; first failure candidates = in-trace unaligned slice/concat and the uint32 id-row reshape, fallbacks in `_agents/mtp_steps345/STATUS.md` |
| B5 DSA decode state fusion | `GLM53_DSA_DECODE_STATE=1`: one `dsa_decode_state` program (q_a norm, latent append, ring + pool in place) replaces the 29 composite state ops in the traced sparse decode, `q_resid @ W_qb1` / `o_packed @ W_o2` absorb folds, free head-major index-query view (`tt/indexer.py` `free_view`); static op count per DSA layer: replicated 50 -> 15 (the ledger said 14; ~-144 us), partitioned heads2users 71 -> 26 (the full-row q_resid for the head-sharded fold stays composite, 2 ops, until the kernel gets a user offset); `latent` mode = the program for q_a / latent only with the composite index maintenance (fallback); +~48 MB bfp8 fold weights per DSA layer per chip when on; 24 host tests (fused vs composite golden bit-identical at B = 1/4/32/40 over 6 steps, fold algebra, tile-order views, knob plumbing) | `0` | `tests/custom_ops/test_dsa_decode_state.py` + `test_dsa_decode_state_mesh.py -k chip0` on chip 0 (selected ids, ring / pool / latent bit-identical, PCC >= 0.9998), `-k block` (heads2users partitioned, capture + replay), `test_mesh_batch_independence.py` and the block e2e with the knob, the 64 K needle with the knob on the 8x4; first failure candidates: ring / pool last-ulp differences (-> `latent`), program q_resid vs rms_norm near-tie flips (`_agents/dsa_b5/STATUS.md`) |

Honest expectation for the decode step after the validation flips the defaults: 83.1 -> ~78 ms at 32 users (A3 + B7),
plus the EP32 32-user path 71 -> ~60-69 ms with the T<32 reduce-scatter combine; the larger items (A1, MTP and B5 after their device tests) remain.

## 2026-10-27 serving bring-up on the 8x4 (vLLM 0.30 + plugin main)

Smokes r2-r6 (`_logs/mesh/serve_smoke_glm53_*.log`): r2 died at boot (plugin main needs vLLM 0.30: `vllm.entrypoints.launchers`), r3
at the 0.30 runner check (`is_vllm_model`: the adapter's `__init__` must accept `vllm_config`), r4 = the first served run (exact first
tokens, 64 K needles 4/4; the chat endpoint hid thinking text; the 64 K random bench killed the engine), r5 died at the first weight
write (the mesh was wedged by r4's hang: a (2,4) block stage passing after a full-mesh device timeout proves nothing about the other
chips; one reset), r6 = a/b/c PASS + the profile: per decode step `vllm_config.check_row` ~50 ms per live user (Python iteration of a
[32768] block-table row; fixed), replay, `update_decode_positions` (11 uploads x 32 chips), `_read` (32-chip readback); engine step
375 ms vs adapter 156 ms under the 32-request bench, ~30 ms outside the adapter in the 4-request stages (window profile queued). The
64 K random prompt hang reproduced (2/2) with a tt-triage capture: devices 0/4/8/12 in `reader_writer_reduce_affine_transforms` ->
`synchronize_head_stage<4>` -> `ready.wait_min` while the other 28 devices are on the next op (AutoTriage in progress; the ttnn op is
outside this directory). Stage verdicts of the day: B5 off (chip-0 capture-determinism assert + 0.9995 < 0.9998 block gate in both
modes), block knob set a no-op (78.3 vs 78.4 ms), MTP block rerun 2/3 (reject-all commit m=0 ring rows pcc 0.50), prefix-incremental
long test dies in a page-table bound (non-default path), ABLOCK flip waits for the two queued 8x4 gates.

### 2026-10-27 02:00-03:45: hang root cause, served numbers, capacity scaling

AutoTriage of the r6 capture: the KDA chronology scalars (`scalar(49152)`) were allocated AFTER the trace captures and clobbered by
~1200 decode replays; ttKDA then derived an interval longer than `sp_size x local_rows` on SP rank 0, its coordinator waited for
5 arrivals from a 4-worker head and its out-of-table NoC write wedged the chips. Fix: persistent per-mesh `chrono_scalar` pair,
`scalar()` refuses to allocate after a capture, `GLMMesh.allocations_frozen` after every capture. r10 then prefilled the 65 536-token
random prompt through the chunk that hung before; a new hang followed in the first decode step after it (not reproduced by the block
reproducer at 2175 / 2176 / 65514 / 65536 tokens in a fresh slot; slot reuse queued). r8 confirmed the KDA SP prefill L1 window
(1024-row slabs) on the 8x4; r9 showed a 4096 bucket costs the 1 M x 32-slot DRAM headroom (buckets 256/512/1024/2048/16384).
Served step 372 -> 192 ms per token (`check_row` iterated the [32768] block-table row in Python); profile inside the benchmark:
replay 148 + read 26 + positions 13 + ids 2 ms. DSA layer per-op profile at 1 M vs short capacity: 2.94 vs 1.88 ms, all of the
delta in the sparse-decode custom ops (#556 731 vs 139 us, #557 228 vs 73, #560 167 vs 20) and the compact SDPA (73 vs 19):
capacity-scaling loops, being bounded by a per-step live-pool count. KDA layer 1.49 ms at 1 M (short pending). Spec K=3 on the
8x4 at 32 users: lossless, 2.81 accepted drafts, x0.55 with the v1 blend commit.

### 2026-10-27 04:15: serving adds nothing to the device step

`e2e_8x4_B4_cap1m_part` (45 layers, max_seq_len 1 048 576, user-partitioned cache, 128-token prompt): traced decode 152.2 ms =
6.6 tok/s per user, traced TTFT 269 ms = the served replay (148 ms) and the README 1 M row. The served-vs-harness gap (69 ms at a
256-token cache) is capacity scaling inside the model. Bounded sparse decode (agent, `arg/glm53_bounded_sparse_decode`): the
`index_score_decode` scorer looped over W/256 chunks of the capacity (W = 262 400 pools at 1 M, `index_kpool` 4) and the segmented
top-k stage 1 had no per-row bound; both now take a per-chip live-pool bound (`SparseDecodeMeta.live_bound`, granule 512), expected
731 -> ~10 us and 228 -> ~8 us per DSA layer; the pool gather and the compact SDPA do not scale. Block reproducers for the r10
decode hang (2175 / 2176 / 65514+25 / 65536 / slot reuse with a 65514+32 prior request) all pass on 4 layers: the hang needs the
served composition (45 layers, 8x4, 1 M) or the random content; a served 131 K natural-text needle run decides whether long served
prompts work at all.

### 2026-10-27 09:00: the 1 M decode gap is the dropped stacked expert copy

Capacity ladder (45 layers, 4 users per row, 128-token prompt, traced step): 256 -> 70.1 ms, 256 K partitioned -> 70.9 ms,
512 K -> 151.4 ms, 1 M -> 152.2 ms, 1 M with 1 user per row replicated -> 139 ms; caches allocated after every weight -> 152.4 ms
(placement excluded); the sparse-decode bound -> 148.6 ms; the KDA layer does not scale, the DSA layer +1.06 ms (bounded away).
The numerics flip with the step (chain pcc 0.8615 below the threshold, 0.9497 above). Cause: `mesh_model` keeps the stacked
(bank-sharded) expert copy for the decode streaming op only while `MemoryPlan.fits(max_batch, max_seq_len)` with the copy; above
~256 K x 4 (and at 128 K x 32) it is dropped and decode runs the regions per-expert lists (`_stream_serves` False), ~+80 ms over 42
MoE layers. Per-op profiling of multi-layer models at 1 M is impossible on this stack (the 1 M build floods the device profiler;
the host capture tool crashes on 45 layers); the single-layer harness keeps the copy, which is why it never showed the gap.

09:37 CONFIRMED: `e2e_8x4_B4_cap256k_onecopy` (256 K per user, `GLM53_REGIONS_KEEP_STACKED=0`) replays in 148.9 ms with chain pcc
0.949673 = exactly the 1 M step and the 1 M numerics (with the copy: 70.9 ms / 0.861457). The full-wall harness at position 1024
agrees: 73.1 ms at 256 K, 159.2 ms at 1 M (per-layer eager walls dispatch-bound on both). Next: `e2e_8x4_B4_cap1m_bfp4` (do both copies
fit with bfp4 experts?) and the one-copy streaming decode (`GLM53_REGIONS_DECODE`).
### 2026-10-27: regions-sourced streaming decode (device-free, `arg/glm53_regions_stream`, `doc/mesh/regions_decode.md`)

The 1 M step (152 ms vs 70.9 at 256 K, same prompts and users) was the expert path: above the planner's fit point the
stacked stream copy (9.41 GiB per chip) is dropped and decode runs on the regions prefill kernel (+80 ms per step, other
numerics). The regions lists themselves -- per-expert DRAM-interleaved `[4096, 256]` / `[256, 4096]` tensors -- already
sit bank-locally in the stream op's order (tile column `n` of an 8-column tile tensor lives in bank `n`, K-contiguous:
the allocator's page round-robin), so the second copy duplicated bytes that were already in place. `GLM53_REGIONS_DECODE=stream`
(default `off`) makes the stream / dispatch ops read the regions lists (REGIONS_SRC kernel variants, the per-expert
addresses as common runtime args, the down block k-major, the (bank, column) maps re-derived): the same packets from
the same banks at the 8x4 shapes, bit-identical results, no second copy (the one-copy configuration never uploads
the stacked tensors). Expected: 1 M x 4 users 152 -> ~71 ms, served 6.6 -> ~14 tok/s per user; 32 x 1 M on the
regions-sourced dispatch ops. Host tests green (planner vs an independent page map, kernel ABI, plumbing); the
one-chip op A/B and the (2,4) block test (one-copy vs regions fallback vs stacked, bit-identity asserted) are written
and not yet run.

10:22 `e2e_8x4_B4_cap1m_regions_stream` (wt_frozen18 = 0a3feca356a, 1 M x 4 users/row, `GLM53_REGIONS_DECODE=stream`): traced step
75.72 ms = 13.2 tok/s per user, 423 aggregate (was 152.2), TTFT 143 ms for 128 tokens (was 264: the short prefill goes through the
dispatch ops instead of the regions kernel), chain pcc 0.861457 = the stacked-copy value (bit-identical path; the one-chip A/B 4/4
and the (2,4) block S1==T1==T2 passed before it). The 4.8 ms over the 256 K stacked point (70.9) = the unbounded sparse-decode
capacity loops (~3.6 ms, `GLM53_SPARSE_DECODE_BOUND=1` not set in this run) + residual; `e2e_8x4_B4_cap256k_default` measures the
stream vs stacked residual on the 8x4. Default flipped to `stream` (the path is bit-identical to the gated stacked path, so the
stacked path's accuracy gate carries over; `GLM53_REGIONS_DECODE=off` restores the old rule).

10:51 `e2e_8x4_B4_cap256k_default` (wt_frozen19 = 257eb14a907: `GLM53_REGIONS_DECODE=stream` default, no stacked copy built,
`GLM53_SPARSE_DECODE_BOUND=1`): 256 K x 4 users/row traced step 70.33 ms = 14.2 tok/s per user, 455 aggregate (stacked copy,
unbound: 70.85), TTFT 142 ms (146), chain pcc 0.861457 (identical). The stream source costs nothing on the 8x4 (ppb=1: the same
packets as the stacked stream), so the short-context README rows measured with the stacked copy stand under the new default; the
1 M residual over 256 K (75.7 unbound -> ~72 bound vs 70.3) is the attention/index capacity term. The stacked copy (9.41 GiB per
chip) is no longer built in any configuration unless `GLM53_REGIONS_DECODE=off`. boundary65536_reuse_8x4_45L (the full-model
reproducer of the served random-65536 decode hang) PASSED at 10:33: the hang stays unreproduced outside the served path.

11:27 serve_smoke_glm53_r12_packed (tree = wt_frozen10 + `GLM53_DECODE_PACKED_INPUTS=1`): PASS a/b/c/d; first tokens exact (both
prompts), determinism 4/4, 4 x 65 514-token needles 4/4 (TTFT 26.5 / 52.3 / 78.3 / 104.4 s, no cross-talk), bench 128 x 128 x 32:
TPOT 174.6 ms = 5.7 tok/s per user (192 before packed inputs: the host share 41 -> ~26 ms per step), 32/32 completed. Packed inputs
default on (mesh_model.packed_inputs_enabled). Next: r13 (+ sparse bound, tree17), r14 (+ stream decode, tree18).

12:05 CORRECTION: r13 (`serve_smoke_glm53_r13_bound`) and r14 (`serve_smoke_glm53_r14_regions_stream`) served wt_frozen10 code: the smoke
script rebuilds the serving tree from `FROZEN` (default wt_frozen10) on every run and `SERVE_TREE` only names the output dir, so the bound
and stream knobs were never read (TPOT 175.2 / 174.8 ms = r12; needles 4/4, first tokens exact, determinism 4/4 on all three). The
script now refuses a `SMOKE_EXTRA_ENV` knob the snapshot does not know. r15 = `FROZEN=wt_frozen20` (stream + packed defaults, fill
clip, + `GLM53_SPARSE_DECODE_BOUND=1`) is the served validation of the streaming decode.

12:21 `long1m_B1_8x4_default` (wt_frozen20: stream + packed defaults, fill clip, `GLM53_SPARSE_DECODE_BOUND=1`; ISL 1 047 612, gen 64, one
prompt replicated to the 8 rows): TTFT 492.4 s (2128 tok/s; the clipped last chunk of 15 420 tokens took 9.2 s vs 7.8 for full chunks),
decode step 73.9 ms = 13.5 tok/s per user at position 1 047 612 (was 148.9 / 6.7). Needle 7/8 rows: row 1 produced "The user asked what
is the secret passphrase for the vault. Answer with the passphrase only." (same first token 785, a near-tie flip later; rows are not
bit-identical per the ring reduce-scatter order) -- watch item if the 4-users-per-row run or the served smokes show more divergence.
README 1 M rows (8 slots / batch 1) refreshed from this run.

12:36 `serve_smoke_glm53_r15_frozen20` (FROZEN = wt_frozen20: stream + packed defaults, fill clip, `GLM53_SPARSE_DECODE_BOUND=1`): PASS
a/b/c/d; the server log shows the one-copy build ("the stacked stream copy is not built", "regions lists usable as the stream source",
ppb 1), boot 616 s (676 with the stacked upload); first tokens exact, determinism 4/4, 4 x 65 514-token needles 4/4 (TTFT 28.1 / 54.6 /
81.1 / 107.6 s); bench 128 x 128 x 32: TPOT 103.2 ms = 9.7 tok/s per user, 37.3 tok/s output (174.8 before), queue TTFT median 48.2 s
(82.5). The served step is now the harness replay (~76 ms) + ~26 ms host; the host share is the next serving lever.

13:17 `long1m_B4_8x4_default` (wt_frozen20 defaults + bound; 4 prompts per row x 8 rows at ISL 1 047 612, gen 160): prefill 484.8 s
(2161 tok/s; per user 485/474/474/472), decode step 78.8 ms = 12.7 tok/s per user, 406 aggregate at 32 slots (was 152.1 / 6.6 / 210).
Needles 30/32: rows 2 and 3 of user 3 produced "The user asked what is the secret passphrase for the vault. Answer with the passphrase
only." -- the same alternative as row 1 of the batch-1 run, always after the shared first token 785. Reading: a near-tie at the second
token under the stream (= stacked) numerics that the row-level ulp noise flips on ~3 of 40 replica slots; the 2026-10-21 runs (32/32,
8/8) decoded on the regions-fallback numerics (chain pcc 0.9497 vs 0.8615) where the tie sat elsewhere. Not a corruption signature
(coherent text, no cross-talk, exact first tokens served); the natural-text logits gates (`logits_8x4_all`) remain the arbiter.
README 32-slot 1 M row refreshed.

13:50 128 K profile boot OOM (`serve_bench_glm53_128k_r1`, 32 users per row): the 32 K warm-up prompt's second 16 K chunk failed a 287 MB
(16384 x 4384 x 4 B: an mHC-width fp32 buffer) allocation with 613 MB free but fragmented; the planner had left 0.55 GiB after the
caches. `reserve_bytes` now takes a 16 K chunk's dense working set (four such buffers, 1.15 GB) as the floor of the prefill transient
(max with the whole-prefix copy and the regions buffers: the observed 1 M points are unchanged), which puts the 128 K profile at
28 users per row (7 slots per chip; 29-31 cost the same 8th slot). Catalog (dev + prod) set to 28 (TIS 08bd78693). The branch carried
two pushes with a failing memory-plan test while the term was being calibrated (d19071b2769, e536042d715); green again at d8b60d56ff2.
Re-queued: `serve_bench_glm53_128k_r3` (128/128 c1 + c28) and `_r4` (65536/128 c28) after the 1 M-profile bench.

14:45 Served targets rewritten per profile (TIS 8d8a6cde7, rows keyed by `impl`): 1 M profile 128/128 c1 TPOT 94.2 ms = 10.6 tok/s per
user (TTFT 272 ms), c4 103.4 ms = 9.7 (TTFT 545 ms mean), 65536/128 c4 TPOT 412 ms = 2.4 (TTFT 68.6 s); 128 K profile (28 users per
row = 224 slots) 128/128 c1 150.9 ms = 6.6 (TTFT 228 ms: every request pays the 224-slot traced step), c28 172.8 ms = 5.8 / 141.8 tok/s
output (TTFT mean 3.3 s), 65536/128 c28 completed 28/28 (the first 64 K random bench since the r10 hang: no hang) at TPOT 2967 ms =
0.34 tok/s per user, TTFT 389 s. The long-context served TPOT is scheduler-bound: the engine runs a step as all-prefill or all-decode
and interleaves 2 prefill steps (16 K chunks, ~7.5 s each) per decode step (plugin `decode_interleave_*` in the tt additional config),
so a decoding request waits ~15 s per token while the other requests prefill. Levers: the interleave ratio (catalog) and per-row
prefill ownership (8 parallel prefills).
14:45 Served 1 M prefill: the diagnosis run (adapter DRAM log) shows the resident DRAM after the warm-up = the planner's weights +
caches (3748 vs 3754 MiB per bank), 2.05 GiB free, no per-chunk leak (<1 MiB per 16 K chunk), the largest free hole shrinking from
289 to 123 MiB per bank over the chunks; the request died at ~196 K on a 201 MB allocation = a [2048, pools] bf16 score block: the
served path ran the scorer's 2048-row default while every harness 1 M run set `GLM53_PREFILL_QBLOCK=512` (1.07 GB vs 0.27 GB at 1 M;
the planner assumed 512). Fix 413b3d9dceb: the score block follows the slot length (2048 up to 256 K, 1024 at 512 K, 512 at 1 M;
the env still overrides), and the planner's reserve uses the same rule. Diag `serve_diag_glm53_1m_qblock` (wt_frozen22, default
path, one 1 M request) queued.

15:37 `serve_diag_glm53_1m_qb256` (wt_frozen24: score block 256 rows at 1 M by default): ONE SERVED 1 M REQUEST COMPLETES: 64 chunks,
TTFT 504.6 s (2076 tok/s), 160 tokens at 95.4 ms per token = 10.5 tok/s per user; DRAM 219 MiB per bank free at the last chunk (largest
hole 161 MiB; the 512-row run had died at chunk 62 on the 264 MB ranking copy). TIS targets gained the 1 M-context row (c1). Next served
point: the profile's cap, 1 M x 4 distinct requests (`serve_bench_glm53_1m_c4`, ~40 min), queued after the 8x4 fused spec stages.

15:46 `spec_8x4_K3_fused_r2` (wt_frozen23: fused multi-step KDA commit, V/C/D graphs traced; 45 layers, 32 users = 4 per row, K=3):
lossless (64/64 tokens identical to plain greedy per user), mean accepted 2.45 drafts = 3.20 tokens per user per iteration, 133 ms per
iteration -> 41.7 ms per token = 24.0 tok/s per user, 768 tok/s aggregate, against the traced plain step's 70 ms (14.5 tok/s per
user): x1.68 (the test prints x5.53 against its eager plain loop). The (2,4) block run had shown the traced fused commit bit-equal to
the eager one. This is the first measured point above 20 tok/s per user; it becomes a README row once a run reports its TTFT with it,
and reaches serving through the adapter's speculative path (the gemma4_mtp pattern: B(K+1) <= 32 per row, greedy only).

16:30 Served speculative decode, code only (branch `arg/glm53_served_spec`, no device run yet): the vLLM adapter implements the
plugin's speculative contract (`docs/SPEC_DECODE_CONTRACT.md`): `spec_plan` (K = min(num_speculative_tokens, `GLM53_VLLM_SPEC_K`
default 3, 32 // users_per_row - 1), narrow decode on, accept `argmax_ids`, the hidden state on device), `decode_forward(spec_mode)`
= graph V over every slot's `[anchor, d_1 .. d_K]` rows (the plugin's placeholder columns carry the anchor), `propose_draft_tokens`
= graph C (the fused KDA / index commit with the runner's accepted counts) + graph D (levels 2..K from the accepted row), or D0 after
an ordinary step; the ordinary decode of a speculating launch is graph P (the plain step + MTP level 1, `SpecDecoder(plain_draft=True)`)
so the MTP cache follows every position; the prefill fills the MTP cache per chunk from the chunk's streams (one extra DSA + MoE layer
per chunk, ~1/45 of the prefill; the last row of an intermediate chunk takes the chunk's greedy chunk-end token). Switch
`GLM53_VLLM_SPEC=1` (off = today's adapter bit for bit) with `GLM53_PARTITION_USERS=0`; `GLM53_SPEC_COMMIT` defaults to `fused` in the
adapter. v1 limits: replicated caches (the user partition is refused: `DsaSpecState`), the dense range (a slot whose anchor + K passes
2050 gets no drafts and its steps are plain; a verify the plugin still sends is answered ONE column wide = the plain step, the drafts
counted rejected: lossless), greedy rows (sampled requests take the plain step through narrow decode), no wave prefill. Host tests
`tests/test_serving_spec_host.py` (13: the K policy, the block translation, the boot order, the per-chunk MTP fill, the served loop vs
the plain chain on a Markov fake with a predictable drafter, the one-column fallback, the contract guards); block test
`tests/test_serving_spec_mesh.py` (the adapter through `initialize_vllm_model` on (2,4) x 4 layers, the runner's calls, tokens vs
the adapter's own plain decode) queued.

16:33 `spec_8x4_K3_fused_ttft` (wt_frozen25): the spec run now reports its TTFT -> README row: 32 users (4 per row), 32-token prompt,
K=3 fused: lossless, 3.20 tokens per user per iteration, 42.0 ms per token = 23.8 tok/s per user (762 aggregate); TTFT 994 ms = the
eager prefill with the MTP hidden capture (3.98 s for the 4 prefills of a row, each serving 8 replicas). Served spec v1 (agent,
arg/glm53_served_spec 0c6ba7898b5 + 99435254322): the adapter implements the plugin's device-propose contract (P/V/C/D/D0 traces,
MTP fill per prefill chunk, narrow decode, argmax_ids accept); limits: replicated caches only (the partitioned 1 M profile is refused),
dense range only (anchor + K <= 2050), greedy only. Block test `served_spec_block` queued (wt_frozen27).

18:52 Dispatch pass budget landed (mesh + upstream, from arg/glm53_dispatch_pass_budget: 617940b91fb, b931230374e, 1cc5eeeadb2). Root
cause of the 17:15 block hang (tt-triage + autotriage, _logs/mesh/triage/AUTOTRIAGE_dispatch_pass_budget_20261027.md): the expert dispatch
ops (33..256 gathered tokens) enumerate passes with a budget P = 8T/32 + E that only holds for <= 8 picks per row; a routing row with far
more non-zeros (NaN / stale rows past seq_len handed to the router by the traced prefill's padding) exceeded it, the down writer clamped
its own loop while the compute consumed unclamped metadata -> deadlock on columns 0-2 (column 3's stale rows differed), the gate_up op
wrote past its L1 buffers. Every passing case before had zero pad rows. Fix: one clamp_passes in every dispatch kernel (flag in table
word 63; the down op sums a token's first 8 contributions in expert order), GLM53_ROUTER_PAD_MASK=1 (rows past seq_len zeroed before
routing, traced prefill carries a persistent [T,32] mask input), GLM53_DISPATCH_GUARD=1 (eager dispatch calls read the counts and raise
naming the call site). Device: one chip over-budget dense+nan RETURN with the flag (2/2), no regression on well-formed routings (4/4);
on the block the guard named the served spec v1's MTP prompt fill (all 128 rows over budget after a traced prefill: a layout bug in that
branch, not the pad rows). Note: the served plain path was exposed too (any prompt shorter than its trace bucket).

20:40 Evening status. Served short prompts validated on the head with the dispatch fix (`serve_smoke_glm53_r16_short`: 40 / 100 / 128-token
prompts x 4 through the padded traced prefill, zero guard trips, 10.2-10.5 tok/s per user). The served speculative path v1's block
failure was the late-collective trap again: the MTP prompt fill at the traced length was the first MTP prefill at that shape and ran
after the captures (fixed on arg/glm53_served_spec_fix b3623daede5: the fill is warmed at every padded chunk length before any capture);
spec v2 (partitioned caches + sparse-range verify, arg/glm53_served_spec_long e520dbf659a) is rebased on it; both wait for their block
runs. The 1 M x 4 concurrent served point: the persistent prefix staging buffer (arg/glm53_prefix_staging, block-validated bit-identical)
plus the dense slabs (arg/glm53_prefill_peak, modelled MoE-phase peak 762 -> 510 MB) survived 100 chunks without an out-of-memory where
staging alone died at chunk 19, but the slabs cost ~6x on the prefill (40-47 s per 16 K chunk vs 7.4 s: eager launch overhead) and the
run hit the 90-minute bound inside request 1; neither is a default. The served 1 M profile stays at two of four concurrent 1 M
requests completing; the next levers are a bfp8 latent cache (halves the 11.7 GiB of caches), the regions MoE output pair (306 MB)
and a 4096-token served chunk, and honest admission (2 concurrent at 1 M) until one lands.

## 2026-10-28 served integration (`arg/glm53_served`: APC + image/video + speculative decode on the base)

21:35 `arg/glm53_served` = the base tip (25806fdef6e: today's defaults, the continued-chunk warm-up, the sealed program cache) +
the feature-only commits of `arg/glm53_served_apc` (9), `arg/glm53_served_image` (7) and `arg/glm53_served_spec_fix` (7), picked
in that order with the base's versions kept for every ported hunk (the auto ladder, the fixed-width page-table views, the every-slot
continued-chunk walk); APC x spec are declared mutually exclusive per launch (the verify path records no content / snapshots).
Host suites (one process per serving suite): the 2026-10-28 default flips had left 24 host tests red on the base itself (the fakes
had no `rewind_user`, the ladder / roofline / dispatch-guard tests encoded the pre-flip defaults): fixed on the base (46d04815672)
and on this branch; all 23 host suites green (394 tests). Device (8x4, 4 layers): image + video (f97) and spec (f96) pass; the APC
test's footprint check (`GLM53_VLLM_APC_ALLOC_CHECK=1`) caught a real bug of the spec convention: keeping only `(logits, ids)`
of the decode capture freed the captured step's streams output under the live trace (-64 KiB per bank right after the baseline;
on the spec branch's own runs a silent hole in the trace's address space). Fixed (1f2f4340629: every captured output pinned).
45-layer served smokes: text on the base tree (1m profile, 4 requests): boot 1066 s (the full ladder + 1317 sealed programs),
logits gate PASS on both references, coherent, decode 73.1 ms per step = 13.7 tok/s per user, traced 128-token TTFT 120-145 ms;
APC on its branch (1m profile): logits gate + coherence PASS, the planner has no DRAM for a device snapshot ring at 1 M (spare
124 MiB per chip vs 141 MiB per entry) so the ring lives in the host store (1 entry per slot). The spec smoke needs the 128k
profile with `max_num_seqs=8` (4 x 1 M replicated caches do not fit; 28 users per row leave no draft rows at K=3). Smoke
additions: stage e (the sample image + its 4-frame clip through `/v1/chat/completions`, PyAV mp4), `NEEDLE_ROUNDS=2` (round 2 of
the 64 K needles = the served prefix-cache hit at 45 layers), frozen snapshots get `bringup/references` from the base worktree
(untracked: the logits gate had no reference on a snapshot). Queued on `wt_frozen98`: the APC test with the default (16 K) and a
32-token hit alignment (an unaligned restore under the sealed cache decides whether option 2 is a default flip), then the three
45-layer smokes (text a-e, spec, APC with the repeated needles).

21:40 APC on the merged tree (8x4, 4 layers, `wt_frozen98`): device store hit 173 / 174 ms (first / warm) vs 338 ms from scratch,
host store 315 / 314 vs 492 ms, both hits bit-identical on the logits and the tokens. The same test with
`GLM53_VLLM_APC_HIT_ALIGN=32` (the 1536 hit restored at the decode-crossing snapshot = an unaligned continued chunk) trips the
sealed program cache in `SliceDeviceOperation` at `TtLatentCache.user_rows` called from `_prefix_rows`: the dense range (absolute
positions below 2048) reads exactly `start` prefix rows and splits the chunk's rows into dense / sparse by absolute position, so
each start is a new program; the sparse path's prefix read is already bucketed (`user_rows_rm`). That is the scope of option 2
(shape-static prefix attention): read the whole dense prefix [0, 2048) with a host-built mask over [start, 2048) in
`_dense_attention`, a fixed dense query-row count per slab (superset rows, `_select_rows`), and the masked SDPA variant warmed per
bucket. Until then the served default restores at chunk-aligned snapshots only (16 K on the served profiles).

### 2026-10-29 00:30: merged-tree smokes, the APC switch, dense-static chip 0, A0 built

- Text smoke on wt_frozen102 (45 layers, 1 M profile, the merged adapter + plugin `arg/stable_prefill_slot`): a/b/c/d PASS
  (twice: a patch of the running smoke script made bash resume at a stale offset and rerun the client; same numbers both
  passes). Step 73.1 ms = 13.7 tok/s per user at 4 live users; bench 128 x 128 x 32 requests: 10.9 tok/s per user, TTFT
  43.1 s mean (8 waves of `max_num_seqs` 4). Image PASS (TTFT 13.6 s); clip = HTTP 400: vLLM 0.30.0
  `get_video_processor_cls_name_from_config` returns transformers 5.20's `{'torchvision': 'Glm5NextVideoProcessor'}` and the
  loader registry's dict lookup raises (upstream main unwraps it). Workaround (host-validated through
  `parse_chat_messages`): the vllm arg `media-io-kwargs: {"video": {"video_backend": "glm5next"}}`; in the next smoke and the
  catalog.
- APC smoke "no round-2 hits": vLLM prefix caching was off. The catalog pins `no-enable-prefix-caching: true` (from before
  APC) and the plugin's capability gate only ever disables, so `GLM53_VLLM_PREFIX_CACHING=1` enabled the adapter's ring while
  vLLM never hashed a block (`enable_prefix_caching=False`, hit rate 0.0 %, round-2 TTFT = round-1; the one "restore" was the
  chunked-prefill continuation at 49152). `serve_smoke_glm53.sh` now flips the vLLM flag with the knob; `serve_smoke_apc_f102_r3`
  (stages a, c x 2 rounds, e with the video backend) is queued after the row-local KDA test. Plan: D7.
- Dense-static prefix attention (A3, `GLM53_DENSE_PREFIX_STATIC=1`) on chip 0, layer 3, three chunk plans
  (1024x3+1025, 320+2048+1729, 320+704+1024+2049): static vs dynamic PCC 1.000000 on all three, both 0.999932 vs HF
  (`test_dense_prefix_static_matches_dynamic`, 6 min). Next: the served unaligned-restore check, then
  `GLM53_VLLM_APC_HIT_ALIGN` 16384 -> 32.
- A0 (ttnn `sequence_parallel_size`, branch `arg/kda_sequence_parallel_size` @089a6daba60, 43 files): built and importing
  after two fixes: the public .cpp of `reduce_affine_transforms` / `affine_exclusive_scan` had the optional before `actual_end`
  while the .hpp (and the symbol nanobind references) had it after (an import-time undefined symbol), and the test's feature
  guard read `inspect.signature` of the `FastOperation` wrapper (`(*args, **kwargs)`): it now reads the binding's
  `__call__` docstring. `kda_local_8x4_sp1` (one prompt per row vs HF per row, cross-row independence, local vs SP) started
  00:27 on the 8x4.

### 2026-10-29 01:00: A0/A1 validated on the 8x4 (row-local KDA prefill), H1 = -5.7 ms per decode step

- `test_mesh_local_kda` on the 8x4 from the base-port override tree (`arg/kda_sequence_parallel_size_base`
  @6abcade6099 = base 73fb055f995 + the ttnn `sequence_parallel_size` + the ttKDA axis fix): one 1024-token prompt per
  row through `prefill_forward(..., local=True)`, layer 0: output PCC vs HF per row 0.999683-0.999952, recurrent carry
  0.99985-0.99987, conv history 0.99999, another row's prompt 0.003-0.006 (independent chronologies); the
  sequence-parallel path on prompt 0 vs the local one: PCC 1.000000. 31 s on the mesh.
- Two defects on the way: (1) the main-based override (origin/main 464267840c3) does not open this Galaxy
  (`Timeout (10000 ms) waiting for physical cores to finish` in initialize_and_launch_firmware, FW bundle 19.16.0),
  while the base runtime opens the same chips before and after -> the override is carried on the base commit for the
  tests here (main keeps `tail`/`init` on summarize_chunk_recurrence where the base has `packed_head`); the
  main-based branch is the PR. (2) ttKDA set the TP collective axis to `None` whenever the chronology was not
  sequence-parallel (a 1D-mesh assumption): size 1 on the 8x4 reduce-scattered the o_proj partials over the wrong
  device set (512 of 4096 columns per chip). The axis now follows the mesh's SP extent; fixed on both branches.
- `e2e_8x4_B4_h1_f103` (`GLM53_HYBRID_GATHER=compact`, 45 layers, 4 users per row = 32): traced decode step 55.80 ms =
  17.9 tok/s per user, 573 tok/s aggregate (61.5 ms before), first token = HF (102647), warm 128-token prompt 897 ms.
  H1+H2 (`allreduce` / `allreduce_ccl`) next in the queue.
- APC r3 analysis and the chip-0 unaligned-continuation test: see the 00:30 entry's follow-up in the plan (D7) and the
  ledger note of 00:59 (`sched_validate.log`).

### 2026-10-29 01:35: H1 + H2 measured (the 20 tok/s per user point), the unaligned-start maintenance finding

- `e2e_8x4_B4_*_f103` (45 layers, 32 users, 128-token prompts, first token = HF in all three): H1 compact gather
  55.80 ms; H1 + H2 all-reduce combine 48.70 ms (20.5 tok/s per user, 657 aggregate); H1 + H2 CCL all-reduce 47.86 ms
  (20.9 tok/s per user, 669 aggregate) against 61.5 ms before. Next: the 2K-prose coherence gate on the combined
  knobs, then the defaults flip and the served A/B.
- The APC r3 artifacts, the real lead: the 4-layer served reproduction (`test_served_vllm_cached_prefix_miss_with_
  budget_split`: B joins with the shared 2048 "computed" and 64 tokens of budget, continues from 2112) stopped at the
  sealed program cache inside `TtDSAIndexer.maintain_prefill`: a start that is 32-aligned but not 128-aligned leaves the
  shape-static maintenance for the OFFSET path (host-int slices = new programs). Served, the seal is off, so those
  programs compile at request time under the live decode traces (the post-capture allocation class of the 2026-10-28
  image hang). The chip-0 layer tests were bit-identical because they ran without live traces. A3's index half (static
  maintenance at any 32-aligned start) is the fix; the rerun with the seal off (numerics under live traces) is queued.
- Waves step A (KDA / dense layers + head) is on the block queue after a prompt-length fix.

### 2026-10-29 01:45: waves step A validated on the block; the APC artifacts = allocations under live traces

- `test_mesh_wave_prefill` (layers 0-2 = KDA + dense, (2,4) block, the override runtime): SP vs a wave on row 0 and on
  row 1: first token equal, logits PCC 0.99998; the two rows' waves bit-identical; two prompts in one wave bit-identical
  to each alone; the idle row's KDA slot untouched; a chunked continuation (row 0 at 64, row 1 at 96) bit-identical to
  the unchunked wave (0.9992 vs the SP chunked prefill); a non-contiguous start refused. 20 s on the block.
- With layer 3 (DSA + MoE, heads2users, 4 users per row): the same checks pass for prompt A (0.99992) and the two-prompt
  wave is bit-identical to each alone, but prompt B's logits land at PCC 0.9973 vs its SP prefill (first token equal).
  A layer-level wave-vs-slab attention test (dense, rows swapped, sparse with per-row starts) is queued to split the
  attention from the MoE and the slot from the prompt.
- APC: the budget-split reproduction with the seal OFF: the adapter's footprint check (`GLM53_VLLM_APC_ALLOC_CHECK`)
  caught the continuation from 2112 allocating a device buffer under the live decode traces (+44.5 KB per bank) = the
  post-capture allocation class. The sealed chip-0 continuation test (queued) lists the programs; the fix is the
  shape-static index maintenance at any 32-aligned start (A3's index half).
- Runtime rule: every wave / row-local test runs from `wt_kda_sp_base` (the ttnn override on the base commit) with the
  model folder overlaid; a frozen tree on the stock runtime fails on `ttKDA(sequence_parallel_size=)`.

### 2026-10-29 01:55: correction on the APC artifacts

The served r3 continuation starts (36736, 52992) are multiples of 128, so the served run stayed on the shape-static index
maintenance with the program-cache seal on (the adapter's default) and compiled nothing at request time; the adapter also
sets `GLM53_PREFIX_BUCKET` at init. The seal-off footprint change and the sealed chip-0 misses belong to the test's own
2112 start (the offset path), an unset bucket outside the adapter and a cold second slot. The round-1 artifacts are open
again; the one served shape the chip-0 tests had not covered, a 3968-row ragged chunk padded to the 16384 bucket and
then continued, is queued (`test_chunked_prefill` served-padded variant). A3's index half remains the enabler for
32-aligned APC restores and wave starts, not the artifact fix.

### 2026-10-29 02:00: the DSA wave path is exact at the layer level

`test_dsa_attention_wave_vs_slab` ((2,4) block, heads2users, layer 3 attention, user 1 = owner column 1): a wave of two
prompts (dump rows [0, 200) on row 0, [200, 500) on row 1, T = 384) vs the slab prefill of each prompt alone: every row
bit-exact, both row orders; the sparse regime in two waves (A[:2048] | B[:1024] at 0, then A[2048:2500] at 2048 |
B[1024:2300] at 1024, T = 1280): bit-exact except 6 of the 452 continued rows of A at 1 ulp (max|diff| 0.002). So the
per-row page-table / pool views, the per-row index maintenance and the static attention with per-row starts are
right; the model-level 0.9973 on prompt B (layers 0-3) is left to the MoE under the per-row pad mask: the layer-3-only
wave test is queued (`prefill_wave` now runs without a KDA layer).

### 2026-10-29 02:25: waves validated on the block, eager and traced (A2 / A4)

`test_mesh_wave_prefill` on the (2,4) block from the override runtime, layers 0-2 (2 users per row) and 0-3 (heads2users,
4 users per row): SP vs a wave on each row (first token equal, logits 0.99998 / 0.99992; prompt B at 0.9973 on 0-3 =
near-tie MoE expert flips from the SP scan's different grouping, accepted by the near-tie gate: layer 3 alone agrees at
0.99997 and the attention wave-vs-slab is bit-exact), the two rows' waves bit-identical, two prompts in one wave
bit-identical to each alone, the idle row's slot untouched, 128-aligned continuations bit-identical to the unchunked
wave, a mixed-start wave (row 0 at 128, row 1 at 0), a non-contiguous start refused; the TRACED waves (captured at 256
and 2048 rows per row in 0.1-0.2 s each) bit-identical to the eager waves, the traced idle-row case leaving the slot
untouched. Also: the served-padded ragged chunk + continuation is bit-exact on the block through the slab path (all
16385 rows), the KDA padded-chunk carry test is queued, and the 4-layer sealed APC reproduction is on its last knob.

### 2026-10-29 02:50: the wave accuracy gate on the 8x4; the served branch re-integrated

- `test_wave_first_token_vs_hf` (45 layers, 8x4, heads2users, 4 users per row, the override runtime): the dump's
  128-token prompt gives the HF first token 102647 through SP, through one wave with the prompt on all 8 rows, on row 0
  with 7 idle rows, and through the captured wave (bit-identical to the eager wave; capture 6.4 s, replay 1.0 s incl.
  the host reads). The 8 rows' logits are not bit-identical (0.93 across rows: the per-row reduce-scatter order) and
  the wave-vs-SP logits sit at 0.95 on this random-token prompt (the SP chain itself is at 0.947 vs HF): the gate is
  the first token per row. Throughput point (cold SP): one wave of 8 x 2048-token prompts 7.1 s vs one 16 K SP chunk
  12.2 s; at 2 K per row the MoE's gathered tokens dominate, so the launch-bound gain is for the short prompts (the
  warm numbers follow).
- The served branch is re-integrated onto today's base: `arg/glm53_served_v2` (worktree `wt_served_rebase`) = the base
  (waves A2-A5, the A0 test tree, A3's dense half) + the served stack (APC, image, spec, the 2026-10-28 integration
  commits) with the nine ports the base already carried dropped. Conflicts: the planner (the wave width next to the
  APC and vision fields: a positional-constructor trap, fixed), the indexer imports, the host ladder test (the base's
  version), the docs (the served lineage). All 378 host tests pass; pre-commit green over the 34 changed files. The
  served 4-layer block tests (APC hit, image, spec) and the budget-split reproduction run from it next.
- The APC budget-split reproduction at 4 layers: A (a normal continuation) is bit-identical to from scratch over 5
  decode steps; B (the cached-prefix miss recompute to 2176, then continued) matches at the first token (0.998) but its
  first decode step is at 0.83 against a clear reference winner (top-2 gap 1.06): a real state difference, not a
  near-tie. B2 (the same chunk plan through plain continuations) is queued to split the miss path from the plan.

### 2026-10-29 03:18: bfp4 gate/up experts rejected on the accuracy gate

`logits_8x4_w13bfp4` (45 layers, 8x4, the S=128 dump): gate/up experts in bfp4 (down bfp8) give a logits PCC of 0.9838
(the 0.98 bar passes) but a top-1 agreement of 83.6 % (bar 85 %) and a top-5 overlap of 88.1 %, against 93.0 % / 95.5 %
with bfp8 experts on the same gate. The PCC bar alone would have let it through; the top-1 gate catches the loss. C12
stays bfp8 (the user rule: 4-bit only when accuracy and PCC are retained); its e2e perf point was dropped as moot.

### 2026-10-29 03:58: the served round-1 artifacts' root cause, fixed; the ttKDA driver vendored

- **Root cause.** ttKDA's sequence-parallel prefill places a chunk on the ranks block-cyclically by the ABSOLUTE
  `actual_start` (`chronology.hpp` `derive`: first rank = start / local_rows mod ranks, a split segment at start mod
  local_rows); the mesh model uploads a chunk as contiguous slabs (rank r = rows [r T/rows, (r+1) T/rows)). The two
  agree only when the chunk start is a multiple of the padded chunk length, so a continuation at 2176 in a 256-row
  chunk (vLLM's budget split on the block) or at 36736 in a 16384-row chunk (the served smoke's round 1) scanned the
  slabs in the wrong order and carried the state at end mod T. The per-component state diff localized it (KDA conv
  taps and recurrent states wrong after the 2176 chunk, every DSA latent row and pool before 2176 identical); the
  adapter's miss-recompute path was exonerated first (B vs B2 at 0.9999). Every earlier chunk test had starts that
  were multiples of the padded length (36736 = 2 x 16384 + 3968: the 32768 start is aligned).
- **Fix (our folder).** `decoder_layer._prefill_forward_impl` hands ttKDA chunk-relative bounds (start 0, end = the
  valid rows): KDA has no positional encoding and the carried state supplies the history, so the position enters
  nothing else; the traced prefill already did this. Confirmed at the layer level (`test_mesh_local_kda`: the
  budget-split plan 1024+1024+128/256+172/256 = the aligned plan against HF on 2348 tokens: last chunk 0.99995,
  recurrent 0.99992, conv 0.99999) and at the served level (`apc_budget_split_block_r10` PASS). The 45-layer served
  APC smoke with the fix is queued.
- **The L1 clash** on the (2,4) block (`static dataflow buffers clash with L1 buffers`, the KDA exclusive scan at
  1024-1088 local rows on 2 ranks; the old served tree and the override runtime fail identically; the 8x4's
  16384-row chunks and <= 512 local rows run): the chunk terms' L1 budget (320 KiB per core) left no room for the
  group summaries plus the scan's circular buffers at that geometry. At the user's direction the ttKDA Python driver
  is now vendored as `tt/ttkda` (from `deepseek_v3_d_p/tt/kda` @ f8fabeb9b9c; the ttnn ops stay shared) and tuned
  there: the budget is 256 KiB, which sends that class to DRAM like the longer chunks. The block served tests keep
  their 2-rank chunks at <= 512 local rows meanwhile (image: chunk 512 + filler 288, PASS; waves: SP chunk 1024 under
  the 2048 wave width, queued); the single-2176-chunk plan of the layer test is the clash's regression check.
- Also: `e2e_8x4_B32_dispatch` (256 users, 45 layers, the 32-row dispatch expert path) 100.9 ms traced step = 9.9
  tok/s per user, 2538 tok/s aggregate, TTFT 135 ms traced, first token = HF (the README's 256-user row).

### 2026-10-29 04:22: the served APC smoke passes at 45 layers with the fix

`served_apc_8x4_fix` (the 1 M profile, APC on, stages a/b/c, wt_frozen117): stage a first tokens = HF (11505 natural
prompt, 102647 dump), stage b's four concurrent greedy requests identical across the two rounds (44 / 64 / 64 / 47
tokens: the round-1 artifacts are gone), stage c's four concurrent 64 K needles retrieved. With the budget-split block
test and the layer test this closes the served APC correctness item; the catalog flip (D7) is unblocked. The vendored
driver's L1 budget was confirmed on the block meanwhile (`kda_unaligned_block_r2`: the 1088-local-row chunk runs, all
plans match HF), and `tt/ttkda` now carries the A0 row-local plumbing, forwarded to the ops only when set.

### 2026-10-29 04:43: the roofline's measured column refreshed; the spec harness's near-tie rule

- `tt/roofline.py` measured entries (32 users) now come from the 2026-10-29 per-layer decode profiles
  (`_logs/mesh/profiler/today_8x4_dsa3` / `kda4`, chip (0,0), one traced replay): the attention rows match the
  2026-10-21/26 profiles within a microsecond; the MoE rows change: the stream expert op 264 us (the default path,
  `Layout.expert_path="stream"`; the 683-us block and the 351-us chip-test projection are history), the hybrid
  gather glue 175 us (x) + 67 us (routing) at their 11-op counts, the padded reduce-scatter 269 us at 32 ops, the
  mHC pre-block 82 us per wrap, the in-trace gaps 68 / 47 us per layer. Per-layer measured totals 1789 (DSA) /
  1375 (KDA) us vs the profiles' 1782 / 1380; the step model 64.9 ms vs the 61.5 ms e2e step at today's defaults;
  floor 29.0 ms (the gap floor counts the glue ops). The ranking is unchanged in substance: H1 + H2 (the glue, 511 us
  per layer measured) first, then the stream op's fixed cost (E1), the mHC pre-block (D13), the gaps (A1).
- `tests/test_mtp_spec.py` lossless check: the 8 rows reduce in different orders and the 45-layer chain amplifies the
  ulp into O(1) margins at a near-tie, so the PLAIN path itself disagreed across rows at token 4 of the sweep anchor
  (rows 2 and 5, margins 0.125 and 2.06, the same two tokens swapped). A spec token the plain path picked on another
  user now counts as that tie; the PERF-SPEC line is printed before the verdict. The K sweep is re-pointed and the
  anchor re-queued.

### 2026-10-29 05:08: H1 + H2 are the decode defaults; the served waves pass on the block

- `logits_8x4_h1h2` (45 layers, the natural 128-token dump, `GLM53_HYBRID_GATHER=compact GLM53_HYBRID_COMBINE=allreduce`):
  logits PCC 0.9952, top-1 93.0 %, top-5 95.5 % = the base's gate numbers exactly, so the compact gather and the
  all-reduce combine are the defaults (abac4d0f93b; the padded path stays behind `padded` / `rs`). Measured with both
  on 2026-10-29: the 32-user traced step 61.5 -> 48.7 ms (20.5 tok/s per user); the e2e stage on the flipped tree
  (`e2e_8x4_B4_h1h2_default`) refreshes the README's 32-user row, and the K sweep runs on the new defaults.
- `served_waves_block_v6` (the override runtime, the vendored driver with the row-local plumbing, the duplicate
  `rewind_user` dropped, the near-tie stop in the step comparison): the served adapter under per-row ownership
  passes its three cases against the SP path from scratch on the (2,4) block at 4 layers: a traced wave of two
  prompts decoded jointly, a prompt beyond the wave width (a traced 2048 wave + the eager continuation wave at the
  width, the sparse rows), vLLM's chunked prefill of one request over two steps beside a fresh prompt. The run
  before it (v5) had the whole served warm-up (both ladders, the waves prepared before and captured after the prefill
  traces, the seal) and case (a) green; its case (b) "failure" was a reference top-2 gap of exactly 0.000 at step 1.
- Next: the 45-layer served run with 32 distinct requests (`served_waves_8x4`: the 1 M profile, `max-num-seqs=32`,
  `GLM53_VLLM_ROWS_INDEPENDENT=1`, stages a-d with the bench at ISL 128 x 32 users), then the catalog (A7).

### 2026-10-29 05:53: the 32-user row on the H1/H2 defaults

`e2e_8x4_B4_h1h2_default` (45 layers, 4 users per row, wt_frozen123 = the flipped defaults): traced decode step 48.36 ms =
20.7 tok/s per user, 662 tok/s aggregate (61.5 ms / 16.3 tok/s on the 2026-10-28 defaults), TTFT 120 ms traced (eager
warm 962 ms), first token = HF: the README's 32-user row. The stage's chain test then failed after its measurement at
the harness's second `capture_prefill` ("KDA scalar(0) would allocate a device constant after a trace capture"): the
harness captures the decode trace before the prefill traces, serving the other way round, and the traced prefill's
constant start was first allocated at that capture; `TtKDA` now allocates it at construction.

### 2026-10-29 06:25: the K sweep's anchor on the H1/H2 defaults

`spec_sweep_8x4_B4_K3_r3` (4 users per row, K = 3, 45 layers, fused commit, wt_frozen125): spec 30.1 tok/s per user at 32
users (964 tok/s aggregate, 33.2 ms per token, 3.37 tokens per user per iteration, 2.47 of 3 drafts accepted), lossless
against plain greedy on all 32 users for 64 tokens (the 10-27 row: 23.8). The harness's own "plain" step (216 ms) reads
the full logits every step for the near-tie gaps; the comparable plain step is the e2e's 48.4 ms, so K = 3 at 4 users per
row is x1.46. The sweep continues over (users per row, K) for the `k_by_rows` table (S1).

### 2026-10-29 07:20: the 32-request served run (three attempts) and the ragged-wave root cause

- `served_waves_8x4` r1/r2 died at the first wave capture: the four SP prefill traces (~480 MB at 45 layers) fill the
  512 MB trace region, and a wave trace of T rows per row costs about the SP trace of 8 T tokens. Under per-row
  ownership the adapter now captures NO SP prefill traces (`GLM53_VLLM_SP_TRACES=1` keeps them): the traced path of a
  fresh prompt is the wave (`GLM53_VLLM_WAVE_TRACE_LENGTHS`, default 256,512), a short lone prompt takes the traced
  wave, a long lone prompt the eager SP chunks (the rows share that work).
- r3 booted (waves 256/512 captured in 21 s, the decode trace with 32 slots): stage a's natural prompt = HF, stage b's
  concurrent greedy rounds identical; stage c retrieved 1 of 4 concurrent 64 K needles (three degenerate answers;
  TTFT 231-511 s, one row per prompt: 32 chunks of 2048 at ~7 s vs the SP path's four 16 K chunks at ~12 s shared by
  8 rows); stage d at ISL 128 x 32 users: TTFT 3.4 s, TPOT 2.0 s (the adapter's own line: 725-1194 ms host wall per
  decode step over 32 live users vs 73 ms at 4), the 64 K x 4 point 1 of 4 within the bench's budget.
- The needle corruption's root cause, caught by the sealed block case (d) (`served_waves_block_v7`:
  `MatmulDeviceOperation: program cache miss` in `moe._mask_pad_rows`): the wave ladder ran FULL waves only, so the
  MoE's pad-row mask program of a RAGGED eager wave (a member's chunk shorter than the width: every prompt's last
  chunk, an idle row) was never built at 2048; in serving (seal off) the needles' last waves compiled it under live
  traces and the replays degenerated. The traced 256/512 waves carry their mask, which is why the short cases passed.
  Fix: every bucket once ragged in the ladder (the SP ladder's b-1 rule); the block case re-runs under the seal (v8),
  then the served run without the APC's host-store snapshots (r5), the suspected per-step cost at 32 slots (35 MiB
  per chip per save, ~1.1 GB device-to-host plus a sync, per slot at every chunk end and decode crossing).
- The dump prompt's first token through a 128-row traced wave differed from the SP path's (98319 vs 102647): the
  45-layer wave gate validated 256 rows, to which a shorter prompt now pads (the smallest captured wave length).

### 2026-10-29 08:50: 16 K needles pass, a wave costs an SP chunk, SP-first placement, the 128 grid

The served 16 K needle stage (4 x 16365 tokens, 4 concurrent, 2 rounds, APC off) retrieved 8/8 with identical texts
across rounds, so the 2-of-4 corruption of r5 needs the 64 K shapes (32-wave spans, the 131072 prefix bucket, 65514
tokens); the harness now replays that exact shape (`test_mesh_wave_prefill.py` long profile: vLLM's chunked-prefill
schedule emulated, 65514 x 4 at 1 M by env, each prompt against its own SP path) at 4 and 45 layers. Its TTFTs settle
the policy: a lone 16 K SP chunk 7.7 s, a two-member wave step of 16 K 56-64 s (eight 2048-row waves at ~7 s: a wave
is a 2048-row forward per row, as is a 16 K SP chunk), so `SlotTable.allocate(prefer_sp)` places an eager prompt on
a slot index live on no row (SP-eligible, up to `users_per_row` concurrent long prompts) and keeps the wave-affine
order for short prompts (a traced wave of eight beats eight traced SP calls). r3's engine death is explained: a
96-token leftover-budget chunk after a 128-token prefix-cache hit ended on the 224 KV block (the plugin's fallback
for a running request short of the 256 grid), and `maintain_prefill_wave` needs every member start on the 128 grid;
`chunk_unit` is now 128 under per-row ownership, the planner refuses an off-grid start naming the cause, and the
plugin branch `arg/prefill_align_wait` (local, pending the go) makes such a request wait for a step that reaches the
grid instead.

### 2026-10-29 08:01: the APC-off 32-request served run (r5)

With the ragged ladder, no SP prefill traces, wave traces 256/512 and the APC off: stage a both prompts = HF (the dump
prompt through the 256-padded traced wave), stage b identical and distinct; the adapter's decode step 58-61 ms host wall
over 32 live users = 16.3-17.5 tok/s per user, 523-549 tok/s aggregate (r3 with the APC's host-store snapshots: 725-1194
ms), so the 32-slot per-step cost was the APC host store (35 MiB per chip per save plus a sync, per slot at every
chunk end and decode crossing); the bench at ISL 128 x 32: TTFT mean 5.2 s (32 arrivals prefilled in waves; the
catalog's `decode_interleave_enabled: false` holds decodes until no continuation is pending), decode 10.6 tok/s per
user by the bench's wall. Stage c: 2 of 4 concurrent 64 K needles; the corrupted two are the smaller-start members of
mixed-start waves whose slots held older, longer content past their prefix (`user_rows_rm` covers the wave's longest
prefix on every row): the block case (d) re-runs over stale slot content. The 64 K x 4 bench completed 4/4 with a
321 s mean TTFT: one row per prompt walks 32 chunks of 2048 at 7-10 s each, against the SP path's 4 chunks of 16 K at
~12 s shared by 8 rows, so long prompts belong to SP whenever a slot index is free on every row; per-row ownership buys
the decode capacity (32 distinct requests) at the cost of long-context TTFT when every index is busy.
