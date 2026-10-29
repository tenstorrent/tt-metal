# DeepSeek-V4.1 on Marko's MoE block (MOE_IMPL=block)

Setting `MOE_IMPL` (env `DS41_MOE_IMPL`, tt/settings.py): `block` (the default since FIX.6: owner decision
2026-10-28, the one configuration is the block MoE with its fused head on the Galaxy's FABRIC_2D_TORUS_XY; flat hangs
on the torus, DS41-130, and the fused head does not build on 8x4 FABRIC_2D lines, DS41-129) or `flat` (the earlier
path, kept selectable) (Marko Staletovic's MoE block, `models/demos/moe_block`, from
`mstaletovic/combine-lean-sender-llk` @ `856d97d5864`). First an A/B on the 2x2. Owner decisions: `reports/moe_block_decisions_2026-10-27.md`
(D1-D10); analysis: `reports/moe_block_for_ds41_2026-10-27.md`.

## What runs

The block graph's step names do not change (reference/ds41_ref.py:block_steps). Per layer, after `ffn_norm`:

| step | flat | block (default) |
|---|---|---|
| ffn_norm | gathers whole rows `[S/SP, H]` (unchanged) | the same |
| router | tt/router.py: whole-row logits (HiFi2, fp32 DEST), fused moe_grouped_topk (ROUTER_IMPL) | no work (`Keep()`) |
| experts | tt/experts.py: column gather, flat op + weighted combine, partial `[S/SP, H]` | **the whole MoE layer**: one `TtMoEBlockLayer` call (tt/moe_block_adapter.py) -> `[S/SP, H/TP]` bf16 |
| shared_expert | tt/shared_expert.py: flat op, bfp8, LoFi | no work (`Keep()`) |
| moe_add | add + reduce-scatter over TP -> `[S/SP, H/TP]` | hands the experts step's output on |
| ffn_residual | mhc_post on moe_add's output | the same |

Inside the block call (ops head, ops tail), per chip `(r, c)`:
1. Router: `mesh_partition` of x to the chip's `H/TP` columns, `ttnn.linear` with the gate's `[H/TP, E]` slice (bf16,
   HiFi4, fp32 out: a partial over H), `ttnn.reduce_scatter` over TP to the token owners (`S/N` rows a chip),
   `ttnn.softplus` then sqrt fused into the bias add (fp32), `combine_rf/router_topk` (top 6 by biased score, weights =
   unbiased scores / sum x 1.5) -> joint routing rows.
2. Shared expert: `ttnn.linear` gate / up on x `[S/SP, H]` with the chip's `I/TP` slice (bfp8, LoFi, fp32 DEST),
   clamped SiLU as 4 ttnn ops, down -> partial `[S/SP, H]`, `fabric_reduce_scatter` by rows to the owners.
3. Owners' rows: `mesh_partition(x, dim=2)` -> `[S/N, H]`.
4. Exchange (`models/demos/deepseek_v3_d_p/tt/moe/tt_k3_token_split_moe.py`, routed-only): routing gather, combine
   planner (deterministic), dispatch prep, arena zero, 2D sparse dispatch, then `flat_routed_expert_with_program`
   (our FlatRoutedExpert's weights, weights_nk) + the weighted fp32 combine in one program, tilize -> owners' sums.
5. Tail: + shared expert's owner rows, `all_to_all_async_generic` over TP -> `[S/SP, H/TP]` (typecast to fp32 under
   GLUE_FOLD=off, as moe_add).

One `MoEBlock` per (mesh, chunk), built at the first MoE call (after the model's load-time L1; its exchange bakes its
arena address), shared by every layer; each layer's weights are a `MoELayer` passed with its call. Router and shared
weights are uploaded at load (`MoELayer.ops()`), so the forward does no host work.

Expert numbering: the block puts expert `b` on chip `b // epc` (row-major); our weights hold checkpoint expert
`gtab[c, r, le]` in slot `le` of chip `(r, c)` (ExpertMapping column-major). The adapter numbers the block's experts by
our slots and permutes the router's rows (gate weight, bias) to match (`expert_permutation`), so the weight cache is
the flat path's.

## The fused head (MOE_HEAD=fused)

Setting `MOE_HEAD` (env `DS41_MOE_HEAD`, read only under `MOE_IMPL=block`): `fused` (the default: owner decisions
2026-10-27 "go with fused head", then fused the default after the A/B below) or `ops` (everything above).
The head becomes one program a chip (tt/fused_head.py HeadSE3): x gathered over TP on its ports straight into the
shared expert engine (bfp8, LoFi, clamped SiLU 10) with its row reduce-scatter to the owners, the router (logits HiFi3
on the engine, fp32 partial sums in a fixed order, sqrt(softplus) + bias, top 6, weights / sum x 1.5), the routing
exchange, the combine planner and dispatch prep. The exchange (dispatch, flat experts + combine) and the ops tail are
unchanged. Planned on the 2x2: engine 10 x 8 cores, 2 row chunks, router on the engine, predicted 584 us a layer.

What differs from the ops head:
- x: ffn_norm keeps the column split `[S/SP, H/TP]` (tt/norm.py `build_norm(gather=False)`, tt/model.py) - the head
  gathers x itself, so ffn_norm's all-gather goes (0.59 vs 1.02 ms a chunk). Same values (the gather is a move).
- Router: Marko's fused router with a new score option (`RouterSpec.score` / `RouterStream(score=)`, default
  `sigmoid`: his models unchanged): `sqrtsoftplus` = `softplus_tile` built with `INP_FLOAT32` (the fp32 polynomial,
  as stock `ttnn.softplus` on fp32 and our `ROUTER_SOFTPLUS=fp32` fork option) then `sqrt_tile`, fp32 in DEST. Its
  bitonic top-K now sorts K's power of two (K 6: the top 8, rows take the first 6; K 4 / 8 / 16 / 32 unchanged).
- Combine order: the head's planner (routing_a2a's) writes no order table, so D7's fixed order is not available.
  The combine snaps each term to 2^-19 instead (Marko's `combine_snap`: exact fp32 sums in any order while sums stay
  under 2^24 x 2^-19 = 32; DS41's routed sums peak at 5.4 on the s4096 golden; per-term error < 1e-6). The block now
  refuses `combine_deterministic` with the fused head.
- Shared expert at LoFi (D5): the engine takes its fidelity as a parameter; `FusedHead.fits` only refused LoFi by
  policy, now accepted. The router's logits stay HiFi3 (fixed in the engine kernel).
- Other changes to Marko's code, defaults unchanged: the engine's down K step `KBD` defaults to the largest divisor
  <= 8 of I's tiles (DS41: 36 tiles a chip -> 6; was a fixed 8 the planner could not fit); routing_stream's X port
  direction on a TP line of 2 (modulo 2 both neighbours looked like +1); the head's outputs kept across forwards with
  the ops tail too (freed, they came back at new addresses between layers and the prepared program was rebuilt every
  layer: clean wall 863 ms a chunk before this fix). Carried as-is from Marko @ 856d97d5864:
  `ttnn/ttnn/operations/matmul_reduce_scatter/` and `ttnn/cpp/ttnn/kernel_lib/perf_instrumentation.hpp`
  (se_rs.py imports its `plan_transport`; Python + JIT kernels, no C++ build).
- Fits the 2x2 as is: 2 links a direction (what the block uses), `l1_small_size` 24576 (arena 1256 KB of the device's
  1279 KB cap).

Measured on the 2x2 (chunk 2048, 8 layers, rung `last`, chunk [53248, 55296); the fused tail's table below adds
repeated wall runs):

| path | device ms / chunk | MoE section ms | clean wall ms / chunk |
|---|---|---|---|
| flat (MOE_IMPL=flat) | 65.74 | - | 138.0 |
| block, ops head | 66.9 | 22.03 | 160.6 |
| block, fused head | 61.97 | 17.45 | 102.2 |

### The fused tail (MOE_TAIL=fused)

Setting `MOE_TAIL` (env `DS41_MOE_TAIL`, read only under `MOE_IMPL=block`): `fused` (default, owner decision 2026-10-27
after the A/B) or `ops` - Marko's TailA2A (models/demos/moe_block/tt/tail_a2a.py): the combine's row-major owner
rows tilized, + the shared expert's owner rows, the TP all-to-all into `[S/SP, H/TP]`, one program a chip (20 worker
cores, the tilize and add under the sends). Its output is the ops tail's (bf16 TILE, DRAM; GLUE_FOLD's typecast after,
in the adapter, as before). Fits DS41 as is (H/TP 80 tiles, 512 owned rows a chip); no code change in the block.
Bit-identical to the ops tail: every one of the ladder's 95 metrics equal.

All on the 2x2, chunk 2048, 8 layers, rung `last` (device: plain profile; MoE = the experts section, the whole block):

| path | device ms / chunk | MoE section ms | clean wall ms / chunk (runs) |
|---|---|---|---|
| flat | 65.74 | - | 138.0 (1 run) |
| block, ops head + ops tail | 66.9 | 22.03 | 160.6 (1 run) |
| block, fused head + ops tail | 61.97 | 17.45 | 102.2, 120.0, 125.2 (3 runs) |
| block, fused head + fused tail | 61.2 | 16.69 | 126.3, 113.2, 110.8, 110.8 (4 runs) |

Every clean wall is the same test and procedure (test_profile.py, BRINGUP_PROFILE_WALL=1: warm-up until a run adds no
program-cache entry - 1 run for every path here - then 10 timed runs with cache misses refused, their median). The
same configuration gives 102-125 ms in different processes, and within a run the 10 timed runs still fall by 10-20 ms
(host-side, not program builds): one sample per path (flat, ops head) is +-10 ms. The paired runs
(walls_ab/: ops tail, fused tail, ops tail, fused tail) put the fused tail ~10 ms a chunk ahead (120.0 / 125.2 vs
110.8 / 110.8). The fused tail's host cost is ~0.5 ms a call (its descriptor cache, keyed on the fresh output's
address, rebuilds on some calls: 17 of 80).

The profiler's real-time mode charges programs by device-operation id, and `run_prepared` launches through
`device_operation::launch`: the head's program shows in the experts section as "(other)" (2 programs a chip,
0.69 ms a layer with the dispatch), as the ops head's prepared exchange launches did (3 a chip, 0.21 ms). So the
"~0.5 ms a layer missing" note below does not hold for this profiler; both heads' numbers include those programs.

Accuracy: block alone (test_moe_block, dsv41_flash-2048) PCC 0.9904 vs host (ops head 0.9903); traced 48 replays PCC
min 0.9903, 0 of 34 same-routing replays differ bitwise; ladder L.s4096 / L.last / L.s55296 pass with every metric
within ~1% of the ops head's (e.g. L.s55296 err_layer_L24 0.0281 vs 0.0280, limit 0.0445). Router unit test
(tests/test_fused_head_parts.py::test_router_sqrtsoftplus, MOE_MESH=2x2): picks equal float64's on every token clear
of noise, 99.4-100% of all tokens, weights within 1.0e-3 (half a bf16 ulp), LoFi shared expert PCC 0.9996. Logs and
scripts: /localdev/dnijemcevic/bringup/fused_head_2026-10-27/.

## Decisions and where they are implemented

| # | Decision | Where |
|---|---|---|
| D1 | A/B on the 2x2 behind `MOE_IMPL` (default `flat` then; `block` since FIX.6, owner 2026-10-28) | tt/settings.py `MOE_IMPL`; tt/model.py `TtDS41Block.__init__` / `_fns` |
| D2 | x reaches the block gathered; ffn_norm unchanged | moe_block.py `MoEOptions.x_gathered` (`_x_shape`, `_head`: no x all-gather, router fed `mesh_partition(x, dim=3)`, x not freed); adapter `block_options()` |
| D3 | Marko's ops router with DS41's gate weight, bias, route scale 1.5, top 6 | moe_block.py `Router` (unchanged); adapter `host()` (bf16 gate, fp32 bias recentred as tt/router.py, rows permuted), `block_model()` (asserts the config's top-k, H, I, route scale, limit against `DSV41_FLASH`) |
| D4 | ops head first; then the fused head, the default (owner 2026-10-27, after the A/B) | adapter `block_options()`: `MOE_HEAD=fused` (default): `head="fused"`, `combine_snap` (section above); `ops`: `head="ops"` (`x_gathered` also requires it) |
| D5 | shared expert LoFi, bfp8 weights | adapter `block_dtypes()`: `MoEDtypes(shared_fidelity=LoFi, shared_activations=gemm_dtype())`; bfp8 weights are the block's default. x / h bfp8 under GEMM_INPUTS=bfp8 (FIX.9, DS41-132: se_engine3 `x_dtype` / `h_dtype`, the ops head's casts) |
| D6 | Marko's expert layout (owner change 2026-10-27, after the flat-path A/B 65.74 vs 68.10 ms / chunk) | DS41's `EXPERTS_LAYOUT` (default `marko`, copied from 979579fee03): tt/experts.py `TtExperts.flat_op` / `flat_layout`; the adapter gives the block that layout (`MoELayer.build(..., layout=)`, new `RoutedExperts(layout=)` so the exchange plans with it) and asserts `marko` equals the block's `RoutedExperts.layout_of`. `legacy` also works (the exchange then plans with LEGACY: untested) |
| D7 | fixed combine order (ops head); superseded for the fused head by the combine snap (owner 2026-10-27: keep the snap) | fused: adapter `COMBINE_SNAP` 2^-19 -> `MoEOptions.combine_snap`; ops: moe_block.py `MoEOptions.combine_deterministic` -> `TokenExchange(deterministic=)` -> `TtK3TokenSplitMoE(deterministic=True)` (the exchange's det planner); adapter `block_options()` |
| D8 | EXPERTS_PRECISION=glm only | adapter `TtMoEBlockLayer.__init__` refuses `full` with the decision in the message (the exchange calls the flat op at LoFi, bfp8 x / h) |
| D9 | carry prepared_program, not the generic_op by-reference change, not mcast_v1 | `ttnn/ttnn/bringup/prepared_program/` (+ 2 lines in bringup_nanobind.cpp, CHANGELOG, README, INDEX.md) |
| D10 | FABRIC_2D lines on the 2x2 | the block reads the open fabric's wraps (`Geometry.of`) |
| D14 | torus allowed (owner 2026-10-27): the adapter no longer refuses a wrapping fabric | untested on DS41 (no torus on the 2x2); the Galaxy bring-up verifies it (Marko ran the block on an 8x4 torus and a 1x4 ring) |

Other changes to Marko's code (defaults keep his behaviour): the exchange's fused experts call passes the op's
`weights_nk` (ours lays out `[N, K]` weights); `MoELayer.build` / `RoutedExperts` take the caller's `layout`;
`test_moe_block.py` has `dsv41_flash` in MODELS and test-only knobs `MOE_X_GATHERED`, `MOE_COMBINE_DET`,
`MOE_SE_FIDELITY`, `MOE_L1_SMALL`, `MOE_FABRIC_PAYLOAD` (test_moe_block and test_moe_block_traced).

## Settings with no effect under MOE_IMPL=block

ROUTER_IMPL, ROUTER_SOFTPLUS (the block uses stock `ttnn.softplus` on fp32), COMBINE_ACC (fp32 accumulators),
COMBINE_PLANNER, EXPERTS_SUM, EXPERTS_XG_PINNED, CCL_IMPL for the MoE collectives (the block's own: stock
reduce_scatter for the logits, fabric_reduce_scatter for the shared rows, all_to_all_async_generic for the tail).
MATMUL_FIDELITY does not reach the router (HiFi4 in the block).

## Numerics that differ from the flat path (judged by the ladder, not by equality)

- Router logits: partial over H/TP (HiFi4) + the CCL's fp32 add (about TF32), instead of the whole row on one chip
  (HiFi2, fp32 DEST); stock softplus instead of the fp32 polynomial. Near-tie picks can move.
- Shared expert: ttnn matmuls with bfp8 weights quantized from `W^T` (16-value groups along N; DS41-35: not exact for
  the fp8 values), activation in 4 bf16 ttnn ops; today's flat op quantizes `[N, K]` (groups along K).
- Combine: fp32 accumulators (today bf16), fixed order (as today).

## Untested (nothing here ran on a device)

- The block on a 2x2 (TP 2, SP 2): never run anywhere (Marko: 2x4 lines, 1x4 ring, 8x4 torus).
- `x_gathered`, `combine_deterministic` in the block, `weights_nk` through the exchange, the DS41 expert permutation,
  and the `EXPERTS_LAYOUT=legacy` plan in the exchange (`marko` is the plan the exchange was built with).
- Our device params: `l1_small_size` 24576 (Marko's tests use 0: 24 KB less L1 for the arenas) and fabric payload
  10880 B (the block needs >= 10272).
- 2 links a direction (the exchange's ports are built for 2; the 2x2 has 4).
- (Checked 2026-10-27: does not apply - see the fused head section.) The bring-up profiler on this branch does not wrap
  `ttnn.bringup.run_prepared` (Marko's branch does, in op mode); the real-time mode still charges those programs.
- Component and swap tests (bringup/hooks.py `_device_step_fn`) build the flat modules step by step, so they never
  exercise the block. Covering it needs a framework change: one step `moe` (inputs `ffn_norm`, output `mlp_out`)
  instead of router / experts / shared_expert / moe_add in the block graph (or a hook that lets a model fuse steps for
  swap tests), with its CPU reference the composition of the four. Until then the block is judged by the ladder.
- O.1: the block calls forks without model cases yet (prepared_program, combine_rf's router_topk / dispatch_prep /
  sparse_all_gather_2d / routing_pack / combine options, fabric_reduce_scatter by rows, flat_routed_expert_with_program
  with the exchange's arguments).

## How to run the A/B

Build the worktree (its own venv and caches; never the primary tree's):

```bash
cd /localdev/dnijemcevic/tt-metal2-moeblock
export TT_METAL_HOME=$PWD TT_METAL_CACHE=$PWD/built PYTHON_ENV_DIR=$PWD/python_env
./create_venv.sh --force && ./build_metal.sh
source python_env/bin/activate
```

1. The block alone, Marko's own options first (is the block sound on a 2x2), then DS41's options:
```bash
MOE_MESH=2x2 MOE_EXPERTS=96 MOE_HEAD=ops MOE_TAIL=ops MOE_L1_SMALL=24576 MOE_FABRIC_PAYLOAD=10880 \
  scripts/run_safe_pytest.sh --dev models/demos/moe_block/tests/test_moe_block.py::test_moe_block -k dsv41_flash-2048
MOE_MESH=2x2 MOE_EXPERTS=96 MOE_HEAD=ops MOE_TAIL=ops MOE_L1_SMALL=24576 MOE_FABRIC_PAYLOAD=10880 \
  MOE_X_GATHERED=1 MOE_COMBINE_DET=1 MOE_SE_FIDELITY=LoFi \
  scripts/run_safe_pytest.sh models/demos/moe_block/tests/test_moe_block.py::test_moe_block -k dsv41_flash-2048
```
   Pass: PCC > 0.98 vs the host model on the device's routing, 3 calls. Then the traced test with the same environment
   on `test_moe_block.py::test_moe_block_traced -k "dsv41_flash-2048 and wall"`: 48 checked replays, and with `MOE_COMBINE_DET=1` the
   `MOE_BITS` line should report 0 replays differing bitwise (D7).
2. DS41's ladder with `DS41_MOE_IMPL=block` (the bring-up skill's ladder rungs: L.s4096, L.last, L.s55296), limits as
   today. The default `DS41_MOE_IMPL=flat` must reproduce today's numbers.
3. Profile: plain profile vs P.24 (MoE steps 22.8 of 66.77 ms a chunk); note the run_prepared gap above.
