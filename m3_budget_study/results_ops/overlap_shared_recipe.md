# Shared expert overlapped with dispatch: device recipe

Knobs (both default off; off = the unchanged path):

| env | effect |
|---|---|
| `M3_MOE_OVERLAP_SHARED=1` | the shared expert runs on its own Tensix sub-device while the routed dispatch runs |
| `M3_MOE_FUSE_SHARED_RS=1` | the shared expert skips its own TP reduce-scatter; its un-reduced down-projection partial is added to the routed `post_combine_reduce` output before the MoE's reduce-scatter (exact: the RS is linear). Works with or without the overlap |
| `M3_MOE_OVERLAP_DISPATCH_ROWS=<n>` | rows given to dispatch in the split (default: 1 for dispatch v1, 2 for v2) |

What changes per MoE layer with OVERLAP (tt/mlp.py `_call_scheduled_shared`, tt/moe/tt_minimax_moe.py `forward`):

```
router_topk -> routing_setup -> [v2 prep] -> LOAD manager (drain)
   -> dispatch on sd0 (rows [0, n))  +  shared gate/up/swiglu/down on sd1 (rows [n, 10), 2D matmuls sized to it)
   -> CLEAR manager (drain) -> free the window's deferred buffers
-> experts_mm -> combine -> moe_reduce (+ shared partial if FUSE) -> [shared RS if not FUSE] -> add_shared
```

No collective runs in the window, and nothing either sub-device still reads is freed in it.

Zones: `mlp/dispatch` (includes the load), `mlp/shared_expert` (the matmuls), then with FUSE off a second
`mlp/shared_expert/tp_reduce_scatter` after the MoE. The zone sums are still serial kernel time; the overlap shows up
only in the per-op device timestamps, which `tools/overlap_check.py` reads.

## Before any device work

```
ps -eo pid,etime,cmd | grep -E "claude|tt-smi|budget|pytest|tracy|profile_4x2" | grep -v grep
cat m3_budget_study/results_ops/.lock        # must name owner=vmelnykov-ops-agent and be ours to use
git -C ~/tt-metal log -1 && git -C ~/tt-metal status --short -uno
```

## Device-free checks (no galaxy)

```
source python_env/bin/activate
pytest -q models/demos/minimax_m3/tests/unit/test_shared_overlap_config.py
```

Sub-device split, 2D matmul configs vs the L1 budget, the knobs, and the op order of the window (ttnn calls
recorded): load before dispatch, dispatch on sd0, shared on sd1, no free and no other op inside, clear before the
experts, x freed only after the shared expert, the addend reaching moe_reduce under FUSE.

## Run everything

```
DRY_RUN=1 m3_budget_study/results_ops/batch_overlap.sh          # lists the 15 cases
nohup m3_budget_study/results_ops/batch_overlap.sh > m3_budget_study/results_ops/logs/batch_overlap.out 2>&1 &
```

Order: kv (3 cases, ~3 min each) -> kvg (3, ~2 min) -> wall (6, ~5-8 min) -> prof (3, ~10 min). About 1.5 h. Subsets:
`KINDS="kv" CONFIGS="off ov"`, `ONLY="ovl_wall_ov_w4096"`. Every case: reset, lock + busy check, 1200 s / 300 s watchdog,
one runs.csv row with SHA + env; results under `results_ops/overlap/`.

Stop early if `ovl_kv_ov` fails or hangs: a hang in the window (dispatch waiting on a fabric the shared expert
does not touch) should not happen, but a program on the wrong sub-device fails fast with a TT_FATAL naming it.

### 1. Correctness

a. Packed-path KV dump, off vs on (the gate_merge recipe): `budget_packed.py`, B=2 (W=4096), slot 0 at h=0, slot 1
   at h=16384, layers 0-6.

```
cat overlap/kv_compare_ov.txt overlap/kv_compare_ovf.txt     # COMPARE_SP=2 compare_kv.py kv_off kv_<cfg> 0:2048 1:18432
```

Layers 0-3 must be 1.00000 (dense, and layer 3's own K/V are written before its MoE). Layers 4-6 see the MoE
output; expect >= 0.999: the shared-expert matmuls use different block configs than the auto-picked full-grid
ones (bf16 accumulation order), and FUSE adds shared + routed in bf16 before the RS instead of after it.

b. 56320-token KV vs the golden, same harness as `kv_2x4_L0-6` (`tools/run_kv_4x2.sh`, PROFILE_MESH=2x4,
   chunk 5120, cache 51200):

```
grep "KV PCC vs golden" logs/ovl_kvg_*.log                    # baseline kv_2x4_L0-6: min 0.98722
cat overlap/kvg_compare_ov.txt overlap/kvg_compare_ovf.txt    # profile_4x2.py --compare kvg_off kvg_<cfg>
```

Pass: the golden min PCC within ~0.001 of kvg_off's, and the off-vs-on compare >= 0.999 on every layer.

### 2. Wall A/B

`budget_sweep.py`, W in {4096, 8192}, points h=0 and h=139264, 2 warm-up + 5 timed forwards:

```
grep -h '"kind": "point"' logs/ovl_wall_*_w*.log              # or overlap/budget_runs.csv (budget_collect.py)
```

Compare `wall_ms_median` per (W, h) across off / ov / ovf. Rough ceiling from the P0-A profile at W=4096, h=0:
dispatch ~750 us and shared ~645 us per MoE layer, so at best ~0.6 ms per MoE layer (4 in layers 0-6, ~2.5 ms of
~81 ms) minus two drains per layer. FUSE saves one TP reduce-scatter per MoE layer on top.
Watch the W=8192 warm-up anomaly: if the h=0 point is slow for every config, rerun with
`BUDGET_POINTS=8192:8192,0:8192,139264:8192`.

### 3. Zone profile per config

`batch_p0a.sh` point `w4096_h141312_prose` (h=139264, PREFIX_QUIET, warm point, no prefix drains) with
PREFIX=ovl_<cfg>, then:

```
cat overlap/overlap_off.txt overlap/overlap_ov.txt overlap/overlap_ovf.txt
```

Per MoE layer: dispatch span, shared span, window, overlap (us) and the sub-device ids seen. Baseline (off, P0-A
profile) reads 0 % overlap, dispatch ~750 us, shared ~645 us, window ~3.3 ms (router, routing_setup in between). With
ov: SUB DEVICE ID 0 on dispatch, 1 on the shared ops, overlap close to the shared span, window ~max(dispatch,
shared). If the shared span grows a lot (99 or 88 cores instead of 120, and DRAM bandwidth shared with dispatch) the
gain shrinks; that is what this profile is for. Per-op zones land in `overlap/per_op.csv` for zones_to_per_op-style
comparison with P0-A.

## Dispatch v2 (4x2 on the middle-rows 4x4 sub-torus)

v2 needs an even axis-0 extent >= 4, so not the (2,4) carve. KV check on the torus carve, compared with the
existing `bench/kv_4x2t_v2_L0-6` (golden min 0.99531):

```
cd m3_budget_study/results_ops
RUN_ID=ovl_kvt_v2_ov M3_MOE_OVERLAP_SHARED=1 CFG_DISPATCH=v2 CFG_COMBINE=v2 tools/run_kv_torus.sh
RUN_ID=ovl_kvt_v2_ovf M3_MOE_OVERLAP_SHARED=1 M3_MOE_FUSE_SHARED_RS=1 CFG_DISPATCH=v2 CFG_COMBINE=v2 tools/run_kv_torus.sh
python3 tools/profile_4x2.py --compare bench/kv_4x2t_v2_L0-6 bench/ovl_kvt_v2_ov
```

The split gives v2 rows 0-1 (streams in row 0, untilizers in row 1 for the TILE input) and the shared expert 88
cores. If dispatch_fabric2d refuses the placement ("the worker nearest stream ...'s eth core is ... outside"), the
nearest workers are not all in row 0 on that chip: try `M3_MOE_OVERLAP_DISPATCH_ROWS=3`. Zone profile:
`MESH=4x2 PREFIX=ovl_v2_ov ONLY=w4096_h141312_prose CFG_FABRIC=2d_torus_xy CFG_DISPATCH=v2 CFG_COMBINE=v2
CFG_MOE_TOPOLOGY=ring TT_VISIBLE_DEVICES=... TT_MESH_GRAPH_DESC_PATH=... PROFILE_PARENT_MESH=4x4 PROFILE_SUBMESH=0
M3_MOE_OVERLAP_SHARED=1 ./batch_p0a.sh` (env as in `tools/run_kv_torus.sh`).

## Op-level device test (full 8x4 galaxy)

```
pytest -q "models/demos/minimax_m3/tests/unit/test_ep_moe_vs_ref.py::test_ep_moe_shared_schedule"
```

Random weights, s128: each of overlap / fuse_rs / overlap_fuse_rs vs the torch ref (0.95), vs the default schedule of
the same MLP (0.999) and a repeat after a default forward (0.9999, catches buffer reuse across the window).

## Risks to watch

- Buffer lifetimes: every tensor either sub-device reads in the window is freed only after the clear (deferred list);
  a Python temporary dropped inside the window would also free its buffer. The recorder test covers the forward;
  nondeterminism across repeats (the 0.9999 repeat check, or kv_ov vs a second kv_ov) is the symptom.
- L1: the sub-device matmuls use 2D configs under a 384-tile (768 KiB) CB budget (tt/moe/shared_overlap.py); an
  "L1 buffer clash" error means the budget is too high for this build's L1 layout.
- Program cache: sub-device programs are separate cache entries; the first overlapped forward compiles them (the
  harnesses' warm-up covers it).
- Two drains per MoE layer; if the host is the bottleneck they are cheap, if the device is, each costs a bubble.
- FUSE changes the rounding (one bf16 add before the RS instead of after) and the RS ping-pong semaphore phase; both
  benign, but compare against off, not against old numbers.
