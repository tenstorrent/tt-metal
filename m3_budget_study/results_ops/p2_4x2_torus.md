# P2: one (4,2) on the middle-rows 4x4 torus with dispatch_fabric2d / combine_fabric2d

SP=4 (axis 0) x TP=2 (axis 1), EP=8, 16 experts per chip, layers 0-6 contiguous, prose `longbook_56320`, bf4 experts,
M3_MOE_W_NDSHARD=1 M3_MOE_HYBRID_THRESHOLD=128, harness KV patches of `tools/profile_4x2.py` (2 K/V heads per chip).

## Setup

- Mesh: the 4x4 sub-torus is opened directly. That is `TT_VISIBLE_DEVICES=2,3,6,7,10,11,14,15,18,19,22,23,26,27,30,31`,
  `single_bh_galaxy_subtorus_xy4_graph_descriptor.textproto` and `M3_FABRIC=2d_torus_xy`, the same as the SP=4 study's M4R/M4E3.
  `create_submeshes(MeshShape(4,2))[0]` is then carved from it, giving torus cols 0-1 over galaxy rows 2-5. A new knob does
  this: `PROFILE_PARENT_MESH=4x4` (with `PROFILE_SUBMESH=0`) in `profile_prefill.py` and `tools/profile_4x2.py`. The layer
  split still follows the 8x4 stage 0, i.e. layers [0,15).
- The fabric came up as TORUSXY with intra-mesh degree 4 on all 16 chips.
- `M3_MOE_TOPOLOGY=ring`, copied from M4R/M4E3.
- **`M3_CCL_TOPOLOGY=linear`, not M4R's `ring`.** That knob only selects the fused o_proj matmul+RS on the TP axis (axis 1). On
  this carve axis 1 is 2 chips with no wrap inside the sub-mesh. `linear` also matches P0-A2, so T1 isolates the fabric.
- Configs:
  - **T1**: torus fabric + v1 dispatch/combine (Ring).
  - **T2**: torus fabric + `M3_MOE_DISPATCH=v2 M3_MOE_COMBINE=v2`.
  - **T3**: torus fabric + v2 dispatch + v1 combine. T3 was added because of the KV result below, and only at h≈141k.
- Drivers: `batch_p2t.sh`, which wraps `batch_p0a.sh` with its new `CFG_*` transport overrides, and `tools/run_kv_torus.sh`.
  Rows are in `runs.csv` (block P2T), `p2t_runs.csv` and `per_op_4x2_torus.csv` (run_id prefix p2t1 / p2t2 / p2t3 = config;
  roofline `roofline_ops.js --mesh 4x2`, plus `ag_kv_head_copies` / `ag_kv_native_est` rows).
- The zone-profile method is P0-A2's: PROFILE_SKIP_COMPILE=1, PREFIX_QUIET=1, PROFILE_PREFIX_READ_EVERY=0, WARM_POINT=3,
  level 2, a real prefix, one process per point, a reset before each.
- Every run finished OK: 7 KV runs and 14 profiles, with no hang and no TT_FATAL.
- **Tree note.** During the batch another session edited and then committed model code on this branch (3e392e9, shared-expert
  overlap). Its knobs `M3_MOE_OVERLAP_SHARED` / `M3_MOE_FUSE_SHARED_RS` default off and were unset here. The runs from
  20:54 to 21:07 saw those edits uncommitted, and later runs have SHA 3e392e9. The dense layer and `shared` agree across the
  boundary (for example T2 and T1 dense layer 29.68 / 29.81 ms, against carved 29.69).

## Wrap check: confirmed

- **The op's own ring check passed.** `dispatch_fabric2d` TT_FATALs unless `ccl::is_axis_wrap_wired(axis 0)`, which requires
  a direct eth link between the last and first chip of every column of the sub-mesh
  (`dispatch_fabric2d_device_operation.cpp:101`, `ccl_common.cpp:170`). Its placement also refuses a link that leaves by the
  same eth core in both directions.
- **The times match the short way.** Dispatch v2 is 0.67 / 1.23 ms (W 4096 / 8192, h 141k, worst chip), against carved v1
  1.12 / 2.21 ms and torus v1 1.61 / 3.20 ms. That is -40% / -44% against carved v1 and -58% / -61% against torus v1.
  On a no-wrap carve the op would have refused to run, rather than "silently routing the long way". The long-way behaviour
  seen before was the carved (4,4) of an 8x4 torus fabric.

## KV PCC vs golden longbook_56320 (layers 0-6, 5120-token chunks, 51200 cache, no tracy)

| run | config | L4 V | L5 V | L6 V | min |
|---|---|---:|---:|---:|---:|
| P0-A2 kv_4x2_L0-6 | carved rows 0-3, 1d, v1 | .99738 | .99582 | .99524 | **0.99524** |
| kv_4x2t_v1 | torus, v1 (T1) | .99738 | .99582 | .99524 | 0.99524 (identical) |
| kv_4x2t_d2c1 | torus, v2 dispatch + v1 combine (T3) | .99738 | .99572 | .99516 | **0.99516** (pass) |
| kv_4x2t_d1c2 | torus, v1 dispatch + v2 combine | .99570 | .99031 | .99006 | 0.99006 |
| kv_4x2t_v2 | torus, v2 + v2 (T2) | .99598 | .98833 | .98722 | **0.98722 (fail)** |
| kv_4x2t_v2_r2 | T2 repeat | .99600 | .98880 | .98796 | 0.98796 |
| kv_4x2t_v2_loadstats | T2 + M3_MOE_LOAD_STATS (host sync per layer) | .99596 | .98889 | .98783 | 0.98783 |
| kv_4x2t_v2cast | T2 + M3_MOE_COMBINE_V2_CAST=1 (fix attempt) | .99738 | .96464 | .96567 | 0.96464 |

Layers 0-3 are identical in every run, because the first MoE is layer 3 and it only shows in layer 4's KV.

**Verdict: T2 fails. `combine_fabric2d` is the error source.** `dispatch_fabric2d` is clean.

Per-chunk and per-SP-row PCC of the dumps against the torus-v1 dump (`bench/kv_4x2t_*`) shows two effects:

1. **A small deterministic deviation from combine v2 reading the fused expert op's bfp8 TILE output directly.** Layer 4 V is
   0.996-0.9998 in every chunk. It is identical in both no-cast runs and disappears with the explicit bf16 cast: L4 then
   equals v1 exactly.
2. **A larger error confined to one SP row of some chunks.** It starts at layer 4's MoE and differs from run to run:
   - chunk 0, SP row 3: L5 / L6 V 0.90 / 0.87;
   - with the cast: chunk 9, SP row 2, 0.81.
   A per-layer host sync (load stats) does not remove it. The routing there is very skewed: one hot expert takes 3.5-5k of
   the 5120 tokens, and per-chip load is up to 5x the mean.

The fix attempt, the one obvious knob, did not fix it: the cast only removes (1). This is recorded rather than debugged
further. T2 perf below is therefore marked **not correct yet**. T3 (v2 dispatch only) is the PCC-clean variant.

## Perf at h≈141k (worst chip ms, mean of sparse layers 3-6; dense = layer 1)

| op | W | carved v1 (P0-A2) | torus v1 (T1) | torus v2 (T2)* | torus v2-disp (T3) | [2,4] (P0-A) |
|---|---:|---:|---:|---:|---:|---:|
| dispatch | 4096 | 1.116 | 1.610 | **0.673** | 0.640 | 1.138 |
| combine | 4096 | 2.539 | 2.990 | **2.005** | 2.519 | 2.022 |
| moe_reduce | 4096 | 1.728 | 2.045 | 1.476 | 1.621 | 2.652 |
| ag_kv (with copies) | 4096 | 2.100 | 1.605 | 1.727 | 1.638 | 0.490 |
| sparse layer | 4096 | 14.217 | 14.640 | **13.066** | 13.328 | 14.404 |
| dense layer | 4096 | 29.690 | 29.809 | 29.677 | 29.776 | 26.711 |
| layers 0-6 | 4096 | 146.0 | 148.0 | 141.6 | 142.9 | 137.4 |
| sparse chip-µs / token-layer | 4096 | 27.77 | 28.59 | 25.52 | 26.03 | 28.13 |
| dispatch | 8192 | 2.212 | 3.200 | **1.233** | 1.227 | 2.276 |
| combine | 8192 | 4.879 | 5.697 | **4.132** | 4.852 | 3.808 |
| moe_reduce | 8192 | 3.414 | 4.109 | 3.472 | 3.284 | 5.123 |
| ag_kv (with copies) | 8192 | 2.476 | 1.843 | 2.257 | 1.754 | 0.831 |
| sparse layer | 8192 | 24.386 | 25.638 | **23.138** | 22.986 | 26.995 |
| dense layer | 8192 | 50.991 | 51.222 | 51.263 | 50.963 | 55.121 |
| layers 0-6 | 8192 | 250.5 | 256.4 | 246.9 | 245.5 | 272.3 |
| sparse chip-µs / token-layer | 8192 | 23.81 | 25.04 | 22.60 | 22.45 | 26.36 |

\* T2 fails KV PCC (combine v2). Chip-µs = sparse layer worst ms x 8 chips / W. [2,4] is its prose run. Chip means
(`--stat mean_ms`) show the same picture: sparse layer at 4096 is 14.09 / 14.45 / 12.69 / 13.11 / 14.21 ms.

**Reading it**

- **Fabric alone (T1 vs carved) costs the v1 MoE ops.** dispatch +44%, combine +18%, moe_reduce +18%, norm_ag and attn_rs
  a little. The torus ring helps `ag_kv` and `ag_idx` instead: `high_bw_all_gather` picks ring from the fabric, and ag_kv drops
  -24% / -26%. Net: +3% (4096) / +5% (8192) per sparse layer at 141k. This is the same sign as M4R vs M4A on the SP=4 4x4.
- **v2 ops (T2 vs T1):**
  - dispatch -58% / -61%;
  - combine -33% / -27%;
  - moe_reduce -28% / -16%, as its input lands sooner and more evenly.
  Sparse layer -11% / -10%. **Against carved v1: -8.1% / -5.1%.**
- **T3 keeps most of it.** v2 dispatch alone gives -6.3% / -5.7% against carved v1, and it is PCC-clean. Its combine is v1
  at carved-v1 speed, not T1's slower torus-v1 speed. At W=8192, T3 = T2 within noise, because T3's ag_kv and moe_reduce
  came in lower.
- **Against [2,4]** at 141k, T2 / T3's sparse layer is 9% / 7% faster at W=4096 and 14-15% faster at W=8192. Their dense
  layer is 11% slower at 4096 (the SP=4 ring at 1024 rows per chip, as in P0-A2) and 7% faster at 8192.
- The fused o_proj+RS is not used (Linear), so o_proj / attn_rs are unchanged. T2 / T3 attn_rs eff reads 24% against 45% (T1 38%).
  It is a small op (0.2-0.3 ms) that absorbs the skew of the MoE before it.

## Depth (sparse layer, worst chip ms)

| W | h | carved v1 | T1 | T2* | [2,4] | T2 vs carved | T1 vs carved |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 4096 | 0 | 13.894 | 14.877 | 13.504 | 15.545 | -2.8% | +7.1% |
| 4096 | 141312 | 14.217 | 14.640 | 13.066 | 14.404 | -8.1% | +3.0% |
| 4096 | 548864 | 22.442 | 21.143 | 19.431 | 17.651 | -13.4% | -5.8% |
| 8192 | 0 | 26.231 | 28.137 | 25.597 | 30.326 | -2.4% | +7.3% |
| 8192 | 141312 | 24.386 | 25.638 | 23.138 | 26.995 | -5.1% | +5.1% |
| 8192 | 548864 | 38.160 | 36.671 | 33.770 | 31.433 | -11.5% | -3.9% |

- **The dispatch / combine savings are flat in depth.** v2 dispatch is 0.62-0.67 ms at W=4096 and 1.19-1.24 ms at W=8192 at
  every h.
- **The growing gap with depth is the torus ring gather.** ag_kv at 549k is 8.2 -> 6.4 ms at W=4096 and 12.8 -> 9.8-9.9 ms at
  W=8192, in both T1 and T2. It is a fabric effect, not an op effect.
- **Shallow, the fabric cost dominates.** T1 is +7% at h=0, and T2 recovers only -2.4…-2.8% against carved.
- **At 549k, [2,4] still wins the sparse layer by 9-10%.** ag_kv on [4,2] is 3.4x / 5x the [2,4] value even on the torus,
  with the per-head copies. The chip-µs per token-layer are 37.95 / 32.98 on T2, against 34.48 / 30.70 on [2,4].
- The full per-point tables for all 6 points follow at the end.

## CAL.effs['4x2'] from the torus carve

`tools/make_cal_effs.py --mesh 4x2 --W <W> --prefix p2t2 --csv per_op_4x2_torus.csv [--native]`. The method is the same as in
`goodput_results.md` §1: chip-mean at h=139264, layers 3-6 / layer 1, latency floor, wave factor. roofline_ops.js reproduces
the measured chip-mean sparse layer: 12.70 against 12.69 ms at W=4096, and 22.28 against 22.27 ms at W=8192.

Files are `tools/cal_effs_ours_4x2_{p2t2,p2t2_native,p2t1_native,p2t3_native,p2mix,p2mix_native,p2mix3_native}_w{4096,8192}.json`.
The table shows the native variants for the ops that move. qkv, o_proj, experts, indexer, sparse, router and misc are
unchanged within 1 point.

| op | W | carved v1 (P0-A2) | T1 | T2* | T3 |
|---|---:|---:|---:|---:|---:|
| dispatch | 4096 / 8192 | 27.3% / 27.1% | 18.6% / 18.5% | 53.2% / 54.2% | 54.7% / 54.2% |
| combine | 4096 / 8192 | 33.2% / 34.1% | 26.8% / 27.9% | 41.9% / 40.5% | 32.7% / 34.3% |
| moe_reduce | 4096 / 8192 | 16.1% / 16.1% | 13.7% / 13.6% | 19.3% / 16.7% | 16.2% / 15.9% |
| ag_kv (native) | 4096 / 8192 | 41.1% / 37.1% | 64.9% / 57.5% | 62.9% / 51.2% | 67.1% / 66.8% |
| ag_idx | 4096 / 8192 | 46.8% / 46.8% | 94.8% / 93.8% | 100% / 100% | 100% / 100% |
| norm_ag | 4096 / 8192 | 54.2% / 47.3% | 45.4% / 40.8% | 51.4% / 46.0% | 51.3% / 46.1% |
| shared | 4096 / 8192 | 49.3% / 53.7% | 46.7% / 51.1% | 41.5% / 45.0% | 41.6% / 45.2% |
| attn_rs | 4096 / 8192 | 45.4% / 43.0% | 38.0% / 36.6% | 24.2% / 24.3% | 24.6% / 24.6% |

## Goodput: 16x[2,4] (our effs) vs 16x[4,2] (torus effs)

Pavlo's sim runs through `tools/run_goodput.js` with our pipe.ringC. The runs are `tools/goodput_p2.sh`, output in
`goodput/p2*_w*.json`. The carved rows are the P0-A2 rows of `goodput_results.md`. The [2,4] column is the same in every row:
29.7k / 24.0k / 30.4k / 20.8k (near) and 40.6k / 31.2k / 42.0k / 28.9k (full) at 4096-10 s / 4096-3 s / 8192-10 s / 8192-3 s.

[4,2]/[2,4]:

| [4,2] calibration | near 4096 10 s | near 8192 10 s | near 4096 3 s | near 8192 3 s | full 4096 10 s | full 8192 10 s | full 4096 3 s | full 8192 3 s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| carved v1, with copies (P0-A2) | -2.4% | -3.3% | -13.3% | -57.7% | +3.2% | +8.1% | +18.4% | +12.9% |
| carved v1, native (P0-A2) | -0.5% | -0.1% | -10.3% | -26.1% | +3.9% | +8.7% | +15.2% | +19.3% |
| torus T1 (v1 ops), native | +0.3% | +1.1% | -3.8% | -0.1% | -0.5% | +3.5% | +3.7% | +4.3% |
| **torus T2, with copies*** | +4.1% | +1.9% | +3.5% | -4.1% | +8.5% | +11.0% | +21.5% | +14.3% |
| **torus T2, native*** | **+8.0%** | **+5.7%** | +9.2% | +10.5% | +8.6% | +11.2% | +27.1% | +12.2% |
| **torus T3 (v2 dispatch only), native** | **+6.0%** | **+7.2%** | +8.0% | +15.9% | +6.0% | +10.1% | +16.5% | +14.9% |
| mixed 2 carved + 2 T2, with copies | -0.3% | -0.2% | -8.5% | -26.1% | +5.6% | +9.7% | +19.0% | +10.8% |
| **mixed carved + T2, native** | **+3.5%** | **+3.1%** | +5.5% | -0.2% | +6.0% | +10.0% | +17.9% | +11.2% |
| **mixed carved + T3, native** | **+2.8%** | **+2.8%** | +1.3% | +2.5% | +4.5% | +9.5% | +21.8% | +10.8% |

Absolute 16x[4,2] goodput, near, 10 s, W 4096 / 8192:

| calibration | goodput |
|---|---|
| T2 native | 32.0k / 32.2k |
| T3 native | 31.5k / 32.6k |
| mixed T2 native | 30.7k / 31.4k |
| mixed T3 native | 30.5k / 31.3k |

**The all-torus rows are an upper bound.** Every one of the 16 [4,2] stages would need an axis-0 wrap. A galaxy has one
middle-rows 4x4 torus, so only 2 of its 4 [4,2] tiles can use the v2 ops. The rows 0-3 / 4-7 tiles have no 3<->0 / 7<->4 link.

**"Mixed" rows.** The sim takes one effs set per mesh, not per stage. So `tools/mix_effs.py` blends the carved-v1 set and the
torus set 50/50 per op as 1 / sum(w/eff), which is the eff of the mean op time. That models a pipeline whose layers are
rebalanced across the two stage kinds, so that fast and slow stages finish together. For layer times this close, the
harmonic mean of per-stage speeds differs from that by <0.2%.

The mix assumes:

- **Rebalancing.** Without it, the non-torus stages set the pace and the result equals the carved row.
- **Per-stage fabric.** The non-torus stages keep the carved 1d-fabric effs. One galaxy-wide 2D torus fabric would put them
  at T1-like v1 costs instead: dispatch +44%, combine +18%, but a faster ag_kv (T1 native ≈ carved native +0.8-1.2 points).
  A runner with a per-stage fabric does not exist today.

## Decision rule (b)

Rule (b): [4,2] must be >= +5% at the SLO with **near** features.

- **All-torus upper bound: passes.**
  - With v2 dispatch + combine and a native gather: +8.0% / +5.7% at 10 s. But combine v2 fails KV PCC today.
  - With the PCC-clean T3 and native: +6.0% / +7.2% at 10 s, and +8% / +16% at 3 s.
  - With the per-head copies (today's harness): T2 is only +4.1% / +1.9%, so it fails.
- **What a real galaxy can deploy (mixed, 2 of 4 stages on the torus): fails.**
  - native: +3.5% / +3.1% (T2), +2.8% / +2.8% (T3) at 10 s;
  - with copies: -0.3% / -0.2%.
  - Full stack is +4.5…+10% at 10 s, which is not usable under (b'): `msa` and `async` are not committed.
- **(c) is unchanged: it still fails.** The runner / MGD / KV-migration pieces for TP=2 are not scoped (goodput_results.md).
  A mixed layout adds two more runner requirements: per-stage fabric (1d for the rows 0-3 / 4-7 tiles, 2d_torus_xy for the
  middle tiles) and non-uniform layer counts.

**So the torus ops move [4,2] from ≈ -0.3% to ≈ +3% (native, mixed), but not to +5%. The recommendation stays [2,4].**

The torus v2 ops would only change it if two things hold. First, the [4,2] stages can all sit on wrapped rings, which a
single galaxy cannot provide; a multi-galaxy [4,2] placement along the 8-row wrap does not give 4-row rings either. Second,
the combine v2 bug and the native multi-head gather (kv_4x2_status.md step b) are both fixed.

## Follow-ups

- **combine_fabric2d on (4,2) EP=8 with skewed routing: two effects.**
  - (1) The bfp8-TILE-input path deviates from the bf16 path (L4 V 0.996 vs 1.0).
  - (2) A run-to-run varying error confined to one SP row of a chunk (PCC down to 0.81).
  - Repro: `RUN_ID=x CFG_DISPATCH=v1 CFG_COMBINE=v2 tools/run_kv_torus.sh`; compare the dumps chunk-wise against `bench/kv_4x2t_v1_L0-6`.
  - The SP=8 E3 check (longbook_5120, one chunk, EP=32) did not catch it.
- **v2 dispatch alone is clean and gives most of the gain.** It is the safe choice for any torus stage.

## Per-point tables (worst chip, `tools/p2_torus_tables.py`)

**W=4096, h=0** (worst_ms, sparse ops = mean of layers 3-6)

| op | carved v1 | torus v1 (T1) | torus v2 (T2) | torus v2-dispatch (T3) | [2,4] |
|---|---:|---:|---:|---:|---:|
| dispatch | 1.059 | 1.533 | 0.624 |  | 1.126 |
| combine | 2.339 | 2.789 | 1.876 |  | 1.859 |
| moe_reduce | 1.541 | 1.863 | 1.450 |  | 2.367 |
| experts | 2.657 | 2.650 | 2.656 |  | 2.649 |
| shared | 0.568 | 0.592 | 0.659 |  | 0.978 |
| ag_kv | 0.289 | 0.258 | 0.423 |  | 0.209 |
| sparse layer | 13.894 | 14.877 | 13.504 |  | 15.545 |
| dense layer (1) | 4.038 | 4.092 | 4.219 |  | 5.061 |
| layers 0-6 sum | 67.685 | 71.864 | 66.820 |  | 77.271 |
| sparse chip-us / token-layer | 27.14 | 29.06 | 26.38 |  | 30.36 |

**W=4096, h=141312** (worst_ms, sparse ops = mean of layers 3-6)

| op | carved v1 | torus v1 (T1) | torus v2 (T2) | torus v2-dispatch (T3) | [2,4] |
|---|---:|---:|---:|---:|---:|
| dispatch | 1.116 | 1.610 | 0.673 | 0.640 | 1.138 |
| combine | 2.539 | 2.990 | 2.005 | 2.519 | 2.022 |
| moe_reduce | 1.728 | 2.045 | 1.476 | 1.621 | 2.652 |
| experts | 2.738 | 2.743 | 2.685 | 2.745 | 2.759 |
| shared | 0.560 | 0.599 | 0.662 | 0.659 | 0.972 |
| ag_kv | 2.100 | 1.605 | 1.727 | 1.638 | 0.490 |
| sparse layer | 14.217 | 14.640 | 13.066 | 13.328 | 14.404 |
| dense layer (1) | 29.690 | 29.809 | 29.677 | 29.776 | 26.711 |
| layers 0-6 sum | 146.013 | 148.047 | 141.563 | 142.916 | 137.428 |
| sparse chip-us / token-layer | 27.77 | 28.59 | 25.52 | 26.03 | 28.13 |

**W=4096, h=548864** (worst_ms, sparse ops = mean of layers 3-6)

| op | carved v1 | torus v1 (T1) | torus v2 (T2) | torus v2-dispatch (T3) | [2,4] |
|---|---:|---:|---:|---:|---:|
| dispatch | 1.078 | 1.567 | 0.628 |  | 1.155 |
| combine | 2.373 | 2.805 | 1.965 |  | 1.838 |
| moe_reduce | 1.569 | 1.895 | 1.636 |  | 2.445 |
| experts | 2.673 | 2.672 | 2.706 |  | 2.685 |
| shared | 0.565 | 0.597 | 0.661 |  | 1.002 |
| ag_kv | 8.195 | 6.433 | 6.369 |  | 1.849 |
| sparse layer | 22.442 | 21.143 | 19.431 |  | 17.651 |
| dense layer (1) | 105.485 | 104.678 | 106.028 |  | 90.546 |
| layers 0-6 sum | 406.336 | 399.199 | 393.903 |  | 342.209 |
| sparse chip-us / token-layer | 43.83 | 41.29 | 37.95 |  | 34.48 |

**W=8192, h=0** (worst_ms, sparse ops = mean of layers 3-6)

| op | carved v1 | torus v1 (T1) | torus v2 (T2) | torus v2-dispatch (T3) | [2,4] |
|---|---:|---:|---:|---:|---:|
| dispatch | 2.163 | 3.110 | 1.235 |  | 2.256 |
| combine | 4.678 | 5.599 | 4.204 |  | 3.735 |
| moe_reduce | 3.184 | 3.946 | 3.234 |  | 5.032 |
| experts | 4.195 | 4.195 | 4.205 |  | 4.174 |
| shared | 1.011 | 1.053 | 1.181 |  | 1.846 |
| ag_kv | 0.835 | 0.730 | 1.100 |  | 0.713 |
| sparse layer | 26.231 | 28.137 | 25.597 |  | 30.326 |
| dense layer (1) | 8.130 | 8.281 | 8.504 |  | 10.912 |
| layers 0-6 sum | 129.573 | 137.569 | 128.178 |  | 153.979 |
| sparse chip-us / token-layer | 25.62 | 27.48 | 25.00 |  | 29.62 |

**W=8192, h=141312** (worst_ms, sparse ops = mean of layers 3-6)

| op | carved v1 | torus v1 (T1) | torus v2 (T2) | torus v2-dispatch (T3) | [2,4] |
|---|---:|---:|---:|---:|---:|
| dispatch | 2.212 | 3.200 | 1.233 | 1.227 | 2.276 |
| combine | 4.879 | 5.697 | 4.132 | 4.852 | 3.808 |
| moe_reduce | 3.414 | 4.109 | 3.472 | 3.284 | 5.123 |
| experts | 4.238 | 4.235 | 4.200 | 4.232 | 4.237 |
| shared | 1.006 | 1.046 | 1.193 | 1.183 | 1.829 |
| ag_kv | 2.476 | 1.843 | 2.257 | 1.754 | 0.831 |
| sparse layer | 24.386 | 25.638 | 23.138 | 22.986 | 26.995 |
| dense layer (1) | 50.991 | 51.222 | 51.263 | 50.963 | 55.121 |
| layers 0-6 sum | 250.506 | 256.350 | 246.872 | 245.548 | 272.310 |
| sparse chip-us / token-layer | 23.81 | 25.04 | 22.60 | 22.45 | 26.36 |

**W=8192, h=548864** (worst_ms, sparse ops = mean of layers 3-6)

| op | carved v1 | torus v1 (T1) | torus v2 (T2) | torus v2-dispatch (T3) | [2,4] |
|---|---:|---:|---:|---:|---:|
| dispatch | 2.082 | 3.052 | 1.194 |  | 2.206 |
| combine | 4.630 | 5.389 | 3.974 |  | 3.620 |
| moe_reduce | 3.068 | 3.583 | 3.193 |  | 4.812 |
| experts | 4.081 | 4.093 | 4.102 |  | 4.109 |
| shared | 1.020 | 1.073 | 1.217 |  | 1.881 |
| ag_kv | 12.825 | 9.873 | 9.827 |  | 1.942 |
| sparse layer | 38.160 | 36.671 | 33.770 |  | 31.433 |
| dense layer (1) | 183.087 | 182.127 | 182.099 |  | 187.930 |
| layers 0-6 sum | 696.708 | 690.714 | 679.490 |  | 684.768 |
| sparse chip-us / token-layer | 37.27 | 35.81 | 32.98 |  | 30.70 |
