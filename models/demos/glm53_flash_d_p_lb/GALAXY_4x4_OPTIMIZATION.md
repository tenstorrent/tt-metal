# GLM-5.3-Flash on a Galaxy 4x4: where the time goes, what to optimize

Branch `mstaletovic/galaxy-unified` at `4086ed23327` (KDA fork, full-mesh MoE input) plus the 4x4 work (see
`GALAXY_RUNBOOK.md` section 1b). Mesh: rows 2-5 of the 8x4 Galaxy as a 4x4 torus (`FABRIC_2D_TORUS_XY`), SP 4 x TP 4,
EP 16 (18 experts per chip). Fake weights, chunk 5120 at 51200, device time of the busiest chip
(`tests/test_layer_perf.py`, program real-time profiler).

## Current numbers

| per-layer device ms        | 2x4 ring (LoudBox-equivalent) | 4x4 torus XY |
|----------------------------|------------------------------:|-------------:|
| kda_dense (3 layers)       | 9.03                          | 5.64         |
| dsa_moe (11 layers)        | 15.64                         | 9.80         |
| kda_moe (31 layers)        | 9.92                          | 6.83         |
| whole-model estimate       | 506.8 ms / chunk              | **336.3 ms / chunk** |

Eager wall time on the 4x4 is 2-3x device time (13-23 ms vs 6-10 ms per layer): the 4x4 only reaches these numbers
traced.

## Profile (4x4, whole model = representative layer x layer count)

| op                              | ms / chunk | share |
|---------------------------------|-----------:|------:|
| `bringup.fabric_reduce_scatter` | 84         | 25%   |
| `linear` + `minimal_matmul`     | 56         | 17%   |
| `bringup.flat_routed_expert`    | 34         | 10%   |
| `bringup.sparse_sdpa`           | 27         | 8%    |
| `bringup.fabric_all_gather`     | 21         | 6%    |
| `to_layout`                     | 16         | 5%    |
| `bringup.moe_ag_local_reduce`   | 13         | 4%    |
| `experimental.all_to_all_async_generic` | 11 | 3%    |

Per MoE layer (both block types alike), the reduce-scatters are: shared expert 0.87 ms (2 calls), routed experts
0.88 ms (2 calls), attention 0.18 ms (1 call). The shared expert's matmuls are only 0.44 ms: it spends twice as long
moving its partials as computing them.

| step (per layer)          | dsa_moe | kda_moe |
|---------------------------|--------:|--------:|
| attention                 | 4.46    | 2.44    |
| experts                   | 2.58    | 2.62    |
| shared_expert             | 1.38    | 1.39    |
| indexer                   | 0.89    | -       |

## What to optimize, in order

1. **Shared expert without reduce-scatters** (est. -35 to -45 ms / chunk, ~12%). Today it is TP-sharded: it computes
   partials on the gathered tokens and reduce-scatters them (0.87 ms of 1.31 ms per MoE layer). For a 4096 -> 2048 MLP,
   replicated weights (~25 MB per chip per layer, ~1 GB for 42 layers, of 32 GB) on each chip's own 320 rows need no
   collective at all (~16 GFLOP per chip per layer). Cheaper variant: add its partial into the routed experts' [T, H]
   partial so both share one reduce-scatter. A model change (`tt/model.py`, the shared expert), no new kernel.
2. **Ring reduce-scatter on the dual Hamiltonian cycles** (est. -20 to -25 ms). The routed experts' reduce-scatter is
   still a line per axis (0.88 ms per layer). `fabric_all_gather` already splits a torus with both sides >= 3 into two
   edge-disjoint Hamiltonian cycles (all four links per chip); a reduce-scatter on the same plan should take ~0.25 ms.
   The full-mesh *line* reduce-scatter (`fabric_reduce_scatter(cluster_axis=None)`, `GLM_MOE_RS_FULL_MESH=1`) only
   reached 0.835 ms (the middle of a 16-chip line carries half of every chip's data) and doubles the bf16 rounding of
   the running sum, so it stays off. With 1's shared-reduce variant this speeds up both experts at once.
3. **Matmul configs for the Galaxy chip** (est. -10 to -15 ms). `minimal_matmul` + `linear` are 56 ms, mostly the KDA
   QKV / output projections (~0.6 ms each). `tt/mm_configs.py` was tuned on the LoudBox's p150 (11 x 10, 640 rows per
   chip); the 4x4 has 4x fewer rows per chip and a 12 x 10 grid. A config sweep first, not a new kernel.
4. **`sparse_sdpa`** (27 ms, 11 DSA layers, 2.5 ms each): the biggest single compute kernel. Check its core use at 320
   queries per chip x 2176 selected keys; it was sized for 640 rows per chip on the 2x4.
5. **MoE layout fusion** (est. -10 to -15 ms): 0.26 ms of `to_layout` (gathered tiles -> row major for the flat
   expert) and 0.30 ms of `moe_ag_local_reduce` per MoE layer. The flat expert could read tiled x / top-k directly, or
   fold the per-token weighted reduce into its y writer.
6. **`all_to_all_async_generic` in MLA** (1.04 ms x 11 = 11.5 ms): the TP head redistribution in DSA attention; a
   different head / row split may shrink it on the 4x4.

Outside the ops: run the 4x4 traced; until then host dispatch dominates and device savings do not show end to end.

## Measured and ruled out

- **120 vs 110 cores per chip.** The Galaxy chip's extra column (column 6) becomes 10 more down cores for the flat
  expert (100 vs the p150's 90 compute cores). `MIMO_FL_SKIP_COLS=6` reproduces the p150 plan on a Galaxy chip:
  - flat bench (one chip, `test_flat_bench.py`): -2.8% to -4.9% at 128-256 tokens per expert, ~0% at 64 and >= 512
    (DRAM-bound below, the 64 fixed gate/up cores above);
  - in the model: `flat_routed_expert` -4% to -5%, whole model -0.75% (363.7 -> 361.0 ms);
  - flat + combine overlap (8x1 Galaxy column): no difference; flat time is unchanged too, so the probe likely does
    not touch that plan (rows-capped plans give the extra column to the gate/up rectangle).
  `MIMO_FL_COLS=11` is not a 110-core emulation on a Galaxy chip (it removes only 2 down cores).
- **Full-mesh line reduce-scatter**: see 2.

## Known issues found along the way

- `flat_schedule_defines` asserts one subgrid for every plan since `de8aaf8023f`, so every two-subgrid flat plan
  (TP4-like shapes) fails on any chip (`test_flat_expert_indexed_shapes` / `test_flat_expert_yrm` `tp4`). GLM's shapes
  are one subgrid.
- `test_flat_routed_expert_op` uses the Python `FlatExpert`, whose `_layout` asserts the p150's 11 x 10 grid; the C++
  op the model uses handles 12 x 10.
- `test_moe_ag_ops_perf[mimo]`: `local_reduce_phase2` 79.1 us on a Galaxy chip vs the p150 band 75.1 us +- 5%.
- Ethernet links between the 4x4's chips and the rest of the Galaxy (physical columns 1 <-> 2) have dropped twice
  (140 -> 124 / 120 links), after partial resets and once without a hang; the 4x4 (and the 8x4 torus) then no longer
  place. `ttop-ipmi-reset` (a full Galaxy reset; it ignores arguments) restores all 140.

## Reproduce

```bash
export TT_METAL_HOME=$PWD PYTHONPATH=$PWD/ttnn:$PWD TT_METAL_PINNED_MEMORY_CACHE_LIMIT_BYTES=0
export TT_VISIBLE_DEVICES=2,6,14,10,3,7,15,11,27,31,23,19,26,30,22,18   # 4x4mid carve, per Galaxy
export TT_MESH_GRAPH_DESC_PATH=$PWD/models/demos/deepseek_v3_d_p/experimental_descriptors/single_bh_galaxy_4x4_torus_xy_graph_descriptor.textproto
export BRINGUP_SPEC=models/demos/glm53_flash_d_p_lb/bringup/spec_galaxy_4x4.yaml
# per-layer device time, with per-op breakdowns inside the given steps
GLM_LP_STEP_OPS=attention,shared_expert,experts,indexer GLM_LP_TOP=20 \
  scripts/run_safe_pytest.sh --no-precompile models/demos/glm53_flash_d_p_lb/tests/test_layer_perf.py::test_layer_perf -s
# flat expert alone, one chip, 120 vs 110 cores
TT_VISIBLE_DEVICES=2 FLAT_BENCH_E=18 FLAT_BENCH_M=64,128,160,256,512,1024 [MIMO_FL_SKIP_COLS=6] \
  scripts/run_safe_pytest.sh --no-precompile ttnn/ttnn/bringup/flat_routed_expert_ttnn/tests/test_flat_bench.py -s
```
