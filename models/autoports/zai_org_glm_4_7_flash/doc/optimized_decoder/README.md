# GLM-4.7-Flash optimized decoder — stage report

Target: `zai-org/GLM-4.7-Flash` (`Glm4MoeLiteForCausalLM`), single Blackhole
p150-class chip (device 0, 1x1 mesh, 11x10 compute grid, 8 DRAM banks),
branch `ttmodelmanager/glm47-flash-probe`.

Implementation: `models/autoports/zai_org_glm_4_7_flash/tt/optimized_decoder.py`
(`OptimizedDecoder`, subclass of `FusedDecoder` with the same public
prefill/decode/paged-cache contract; per-device layouts, program configs and
the precision/fidelity policy are the delta).
Tests: `tests/test_optimized_decoder.py` (all against `OptimizedDecoder`),
`tests/test_optimized_perf.py` (before/after perf, 8 signposted windows),
`tests/test_long_context.py` with `GLM47_DECODER=optimized`.
Probe: `probe/optimize_perf_probe.py` (isolated-op candidate sweeps).
Work log: `work_log.md` (all sweep tables, controls, commands).

## Headline results (same session, fused baseline -> optimized, ctx-1024 decode / S=2048 prefill)

| window | fused baseline | optimized | delta |
|---|---|---|---|
| moe traced decode, batch 1 (wall) | 1.034 ms/tok | **0.491 ms/tok** | -52.5% |
| dense traced decode, batch 1 (wall) | 0.969 ms/tok | **0.447 ms/tok** | -53.9% |
| moe warmed prefill S=2048 (wall) | 210.0 ms | **125.5 ms (16.3k t/s)** | -40.2% |
| dense warmed prefill S=2048 (wall) | 15.2 ms | **10.3 ms (198k t/s)** | -32.2% |
| moe decode device time (tt-perf-report) | 787.1 us/step | **333.7 us/step** | -57.6% |
| moe decode ops per step | 61.2 | 67.9 (added ops are measured winning-candidate glue) | |

Baseline arm = the fused decoder exactly as it shipped (bf16 weights, bf8
experts, bf16 cache). Optimized arm = the deployment policy (bf4 attention +
bf4 shared expert + bf4 experts + bf8 dense MLP + bf8 latent cache + LoFi
decode). Wall
JSON: `perf_wallclock_{fusedbase,opt}_{prefill,decode}_{moe,dense}.json`;
tt-perf-report tables/CSV per window:
`tracy/{moe,dense}/{prefill,decode}_{fusedbase,opt}_perf_report.{txt,csv}`
(signposts `PERF_{PREFILL,DECODE}_{MOE,DENSE}_{FUSEDBASE,OPT}`). The raw
full-session ops CSV stays disk-only under `generated/profiler/reports/`
(repo 500 KB commit limit); the committed per-window CSVs carry every row.

## Precision / fidelity policy (final, per tensor group)

| group | dtype | fidelity | evidence |
|---|---|---|---|
| activations / residual / norms | bf16 | norms HiFi4+fp32acc | unchanged from fused |
| attention weights, decode (wqkv_a_ds, wq_b_ds, wo_ds, w_uk, w_uv) | **bf4** | LoFi | OPT-007 trial: real-weight decode PCC unchanged (0.9975 moe / 0.9999 dense), 202k real-weight windows equal to the fused control; 0.513 -> 0.500 ms/tok |
| attention flat prefill copies (wqkv_a, wq_b_heads, wo) | bf8 | HiFi2+fp32acc | prefill numerics headroom; prefill is flash-dominated |
| shared-expert weights | **bf4** | LoFi decode / HiFi2+fp32acc prefill | real-weight trial: moe decode 0.500 -> 0.492 ms/tok (x46 layers at full model); 202k rows-at-bar unchanged vs the fused control (end 24/32 vs 23/32, agg 0.98822 vs 0.98752); decode@202751 0.99714 vs control 0.99778 (the shared-bf4 delta, above the 0.99 bar) |
| dense-MLP weights | bf8 (bf4 REJECTED on measured evidence) | LoFi decode / HiFi2+fp32acc prefill | bf4 dense MLP measured: isolated rows 179 -> 127 us and dense decode 0.449 -> 0.443 ms/tok, BUT the real-weight 202k dense control loses measurably (decode@202751 0.99865 vs 0.99993, end window 28/32 vs 30/32 rows, `long_context_dense_real_expertbf4_cachebf8_mlpbf4.json`) for a ~0.006 ms/tok model-level win on 1 of 47 layers - rejected as an accuracy-for-nothing trade |
| routed experts | **bf4** | LoFi decode / HiFi2+fp32acc prefill | deployment contract (doc/probe/README.md); faster than bf8 at every swept geometry (62 vs 82 us gate_up) |
| router gate | fp32 | HiFi4+fp32acc | selection semantics preserved (tie machinery unchanged) |
| latent KV cache | **bf8** deployment / bf16 supported | - | real-weight 202k: bf8 == bf16 (dense end agg 0.99405 vs 0.99429, both == fused control 0.99395); halves cache to 5.8 GiB at 202k |

Synthetic-weight tests run the comparability arm (bf8 attention + bf8
experts + bf16 cache) at the historical 0.995 bar; gaussian weights lose
about 2x more than the real checkpoint to bf4 block quantization (OPT-012),
so the deployment dtypes are accepted on real-weight evidence (bar 0.99 for
the bf4-expert arm, unchanged from the fused stage).

## What was done (operation-topology audit -> actions)

The audit and every candidate table live in `work_log.md`. Summary of kept
changes (all measured, traced, PCC-checked):

1. **DRAM-sharded decode matmuls** for wqkv_a (2048->1344, N zero-padded to
   1536 for the 8-bank grid), wq_b (768->5120), wo (5120->2048),
   shared_down (1536->2048), dense_down (10240->2048): weights width-sharded
   across the 8 DRAM banks, activations width-sharded in L1 on the matching
   8-core raster, `in0_block_w` swept per role (8/3/10/6/10). Decode copies
   are separate tensors from the prefill interleaved copies (+~0.5 GiB bf4/bf8
   over 47 layers, inside the 32 GiB budget).
2. **Wide-1D mcast decode matmuls** (interleaved weights, qwen36-blackhole
   idiom) where DRAM-sharding loses: shared gate/up (24 cores, SiLU fused into
   the gate matmul), dense gate/up (80 cores).
3. **Absorbed per-head matmuls** (w_uk b20 32x192x512, w_uv b20 32x512x256):
   explicit `MatmulMultiCoreReuseProgramConfig` on a 5x4 grid, one head per
   core — the two worst fused rows (54.1 and 122.1 us at 43-73 GB/s) drop to
   13.1 and 16.1 us.
4. **Sharded RMS norms**: width-sharded L1 residual (8-core grid matching the
   DRAM-sharded matmul in0 raster) + `LayerNormShardedMultiCoreProgramConfig`;
   the 1-core LayerNorm rows (10-35 us each) drop to ~5.4 us.
5. **Router**: tuned 2-core 1D config for the fp32-weight logits matmul
   (25.8 -> ~7.5 us in-layer; 8.5 isolated) keeping fp32 out/acc; the ~37 us single-core `ttnn.gather`
   of routing weights replaced by an embedding-table pick over the bf16 scores
   (~18 us chain) in the batch-1 indexed path.
6. **Routed sparse expert matmuls**: geometry swept across the bf4/bf8 cross
   product; gate_up 8x6-grid pcn2 bw8 osw2 (125.4 -> 48.4 us), down 8x8-grid
   pcn1 bw16 (78.8 -> 31.4 us), L1 outputs (gpt_oss idiom), LoFi.
   Prefill sparse: gate_up bw 8->32 (61.4 -> 37.4 ms per all-ones 1024-token
   chunk), down pcn1 bw24 (37.8 -> 18.3 ms), osw pinned to 1 (op bug below).
7. **L1 residency**: decode residual and MoE glue in L1 (width-sharded
   residual; interleaved-L1 intermediates); kvpe stays DRAM for the
   batch-32 `paged_update_cache` CB headroom.
8. **bf8 latent cache** end-to-end: prefill kvpe typecast to the cache dtype
   before `paged_fill_cache`; decode `paged_update_cache` keeps its bf16
   input; both flash ops read the bf8 cache directly.
9. **SDPA decode config** re-swept under bf8 cache: k_chunk 128 +
   max_cores_per_head_batch 8 stays optimal (k256+ fails CB allocation, mc16
   slower).
9b. **Precision-locked geometry re-sweep at bf4/LoFi** (OPT-014) for every
   row the bf4 policy touches: wqkv bw8 holds (10.3 us; bw4 15.0, 16c 15.1);
   wq_b 8c bw3 holds against the wider-shard 4c bw6 (12.5 vs 12.0 isolated,
   but 4c costs +0.3 us in the coupled 4-core q-norm - whole-path wash,
   8c kept); wo bw10 -> **bw20** (24.6 -> 22.9 us); w_uv bw16 -> **bw8**
   (13.5 -> 12.8); w_uk unchanged (11.5-12.2 across bw2/3/6, kept bw6);
   dense down bw10 -> bw20 in the (rejected) bf4 arm; shared/dense bf4 MLP
   rows measured per the table above. Probe log:
   `probe_results/probe_p1p2_bf4_sweep.log` + work_log tables.
10. **Explicit 2D mcast prefill configs** for the flat prefill projections
   (wqkv_a, wo, shared/dense gate/up/down; SiLU fused into the gate config)
   after the tt-perf-report advice pass flagged their default in0_block_w=1:
   11x10-grid bw8 candidates measured 82->47 (wqkv), 235->134 (wo), 452->247
   (dense down), 91->55 us (shared gate) per 1024-token chunk; out-blocks
   capped (4x8) because per_core-sized CBs clash with resident L1 tensors
   in-layer; small-M chunks (< 10 M-tiles) keep the default config so
   non-aligned short prefills are unaffected. Dense prefill 11.1 -> 10.3 ms;
   the per-head batched prefill matmuls (wq_b_heads/w_uk/w_uv) keep default
   configs — an explicit MatmulMultiCoreReuse candidate needs per_core_M=32
   blocks whose CBs exceed L1 (no out-block fields on that config class to
   cap them). They are ~1.6% of the moe prefill window device time; on the dense window
   they are ~19%, but dense is 1 of 47 layers so the model-level exposure is
   ~0.5% of prefill — recorded as a known limitation with the exact blocker.

## TTNN bug found: sparse_matmul out_subblock_w>1 corrupts multi-group outputs

In the non-indexed sparsity-walk mode, `out_subblock_w>=2` corrupts
multi-group outputs: G=4-group prefill-shaped gate_up at bf4 gives PCC vs
torch 0.9939 at osw=1 (any pcn/bw) but 0.82-0.87 at osw=2/3, worsening with
K-block count. The indexed/gather mode is immune (bit-identical PCC for osw1
vs osw2). Repro script and numbers in `work_log.md`; the sparse host
validation (out_block_w % out_subblock_w etc.) does not catch it. This stage
pins osw=1 on the union (batch>1) and prefill sparse paths and keeps osw=2
only on the immune batch-1 indexed path (58.0 vs 106.8 us).

## Rejected / blocked candidates (each with measurements; details in work_log.md)

| candidate | verdict |
|---|---|
| fold W_UK into wq_b per-head (batched broadcast-LHS matmul) | rejected: only legal path is the non-mcast in0-reuse config; 53-470 us vs ~56 us for the split chain |
| batched-DRAM-sharded absorbed matmuls (pad 20->24 heads) | rejected: 13.8+16.0 us isolated == the integrated reuse-5x4 rows (13.1+16.1) without padded-head plumbing; bf16 variant fails CB-clash |
| batched per-head wo + fast_reduce_nc (replaces concat-heads path) | rejected: 80.8 us vs ~56 us |
| host-folded w_uv@wo per head | blocked: CB overflow (program.cpp:1875) |
| packed shared gate_up | rejected: 26.97 vs 26.75 us separate (SiLU fold lost) |
| packed dense gate_up | rejected: 139.1 vs 137.6 us separate |
| `num_workers_per_dram_bank=2` (BH knob) on wo / shared_down / wqkv / dense_gate | rejected: 38.7/15.9/16.1/96.0 vs 26.0/11.1/10.8/71.2 us |
| bf4 dense MLP (gate/up/down) | measured then rejected: real-weight 202k dense control regression for a negligible model-level win (policy table above) |
| wq_b DRAM-sharded on 4 cores bw6 (wider shards) | rejected: 14.0 vs 13.1 us (8c bw3; K=768=24t caps 8-core bw at 3) |
| dense gate/up DRAM-sharded at bf8 | rejected/blocked: bw>=8 CB-clash, legal configs 87.5+ vs 71.2 us wide-1D (at bf4 the family becomes legal and wins, 42.5 us - used only by the rejected bf4-dense arm) |
| SDPA k_chunk 256/512, max_cores 16 | rejected: slower or CB failure |
| gather on bf16/L1 (instead of fp32) | rejected: 36.6-39.4 us, all equal; replaced by embedding pick |
| LoFi for prefill sparse experts | rejected: no speed change (mcast-bound); HiFi2+fp32acc kept for numerics |
| sparse_matmul osw=2 on union/prefill paths | blocked by the op bug above (osw=1 pinned) |

## Correctness evidence (final code, PCC vs HF fp32 reference layer)

Synthetic comparability arm (bar 0.995; logs/pytest_opt_synth.log, 23 passed
incl. the dense traced-decode PCC test and the batch-8 + bf8-cache cross):

| test | result |
|---|---|
| prefill moe S=17/64/65/512/1024/1057/3000 | 0.99904-0.99948 |
| prefill dense S=17/512/3000 | 0.99996-0.99999 |
| decode moe (prefill 509, 8 steps) | agg 0.99584; pos-509 sub-ulp tie proven (gap 1.1e-3) |
| decode dense | 0.99995 |
| paged cache content vs exact linear ref | 0.99996 |
| bf8-cache arm: prefill / cache content | 0.99924 / 0.99994 |
| batch 8 mixed non-aligned positions (union path) | all users >= 0.9998 |
| batch 32 (union path) | 31/32 >= 0.995, 1 proven tie |
| traced replays + bitwise determinism | 0.9999, bit-identical repeat |
| traced stress: 96 replays, 3 sweeps x 32 positions | all at bar or proven tie; bitwise-repeatable |
| construction audit (DRAM-sharded weights, LoFi cks, tuned sparse configs) | passed |
| runtime no-host-fallback tripwire + static torch-import audit | passed |

Real weights (logs/pytest_opt_real.log, 3 passed):

| arm | prefill S=512 | decode agg (8 steps) |
|---|---|---|
| moe bf8-experts arm | 0.999238 | 0.99697 (pos-514 tie, same as fused) |
| dense | 0.999982 | 0.99994 |
| deployment arm (bf4 attn + bf4 experts + bf8 cache) | 0.995756 | 0.99469 (fused bf4 control: 0.99707/0.99489) |

## Long context (202752 advertised; ladder + controls)

Every evidence JSON carries an `arm` provenance block (decoder, weights,
dtypes, bar) and an arm-suffixed filename so no rerun can overwrite another
arm's artifact. Comparability arm (synthetic, bf8 attention/MLP/experts,
bf16 cache — identical arms to the fused ladder; 5 passed,
logs/pytest_long_synth_attnbf8.log, JSON
`long_context_{moe,dense}_attnbf8_mlpbf8.json`,
`long_context_aligned_202752_attnbf8_mlpbf8.json`):

| 202751-token evidence | dense (control) | moe |
|---|---|---|
| prefill wall | 71 s (2851 t/s) | 86 s (2348 t/s) vs fused 90 s (same bf8-expert arm; the bf4-expert deployment arm prefills 202751 tokens in 51.4 s = 3941 t/s) |
| latent cache PCC vs exact linear ref | 0.999959 | 0.999936 |
| window start / middle / end (rows at bar) | 32/32, 32/32, 32/32 | 30/32+2 expl, 32/32, 28/32+4 expl, 0 unexplained |
| window end agg | 0.999701 (fused: 0.999704) | 0.994927 (fused: 0.994917) |
| decode at position 202751 (full cache) | 0.999951 | 0.999900 |
| aligned S=202752 final window | - | 28/32 + 4 explained, cache 0.999960 |

Deployment-policy real-weight run (bf4 attn + bf4 shared expert + bf4
experts + bf8 cache, bar 0.99; logs/pytest_long_real_deployment.log and a
bit-identical rerun logs/pytest_long_real_deployment_rerun.log, JSON
`long_context_{moe,dense}_real_expertbf4_cachebf8.json`) with the matched
fused controls (logs/pytest_long_real_fused_{dense,moe}.log,
`long_context_{moe,dense}_fusedcontrol_*.json`):

| 202k real-weight row | fused control | optimized deployment |
|---|---|---|
| dense end window agg / rows at 0.99 | 0.993950 / 29/32 | 0.994197 / 30/32 |
| dense decode at 202751 | 0.999992 | 0.999927 |
| moe end window agg / rows at 0.99 | 0.987522 / 23/32 (7 expl, 2 unexpl) | 0.988220 / 24/32 (6 expl, 2 unexpl) |
| moe decode at 202751 | 0.997777 | 0.997138 (shared-bf4 delta; window aggs all above control) |

The optimized deployment policy is equal-or-better than the accepted fused
baseline on every real-weight 202k window aggregate and rows-at-bar count
(individual sub-bar rows differ by <=7e-4 in either direction; the one
visible decode exception is
decode@202751 on moe: 0.99714 vs 0.99778, the shared-bf4 delta, above the
0.99 bar). NOTE: the deployment-arm pytest runs FAIL their 0.99-bar asserts
by design of the ladder (2 failed in both preserved logs) - the failing rows
are moe 101393 @0.9714, 202734 @0.9655, 202745 @0.9412 (a sub-ulp tie whose
reconstruction ceiling under bf4-expert noise is ~0.97) and dense
202719 @0.9873 / 202747 @0.9853; the identical rows fail in the fused
control within ~7e-4 of the deployment PCCs (the control is marginally
higher on the three moe rows - the shared-bf4 delta - and lower on both
dense rows, and it fails one extra row per end window: moe 9 vs 8, dense 3
vs 2), which together with the better aggregates and rows-at-bar is the
acceptance argument. These rows exist identically in the fused control: they are a
property of the real checkpoint at full context (and, for moe, a
reconstruction-machinery ceiling under bf4-expert noise), not an
optimization regression. Decomposition arms rerun on the final code
(all-bf8 weights + bf8 cache: `..._real_expertbf4_cachebf8_attnbf8_mlpbf8`;
deployment weights + bf16 cache: `..._real_expertbf4`; plus the earlier
pre-MLP-policy arms kept as `*_legacy.json`) bracket every dtype axis: the
real-weight arms sit within ~1e-3 of each other and of the fused control on
the same rows. The synthetic ladder at bf4 attention fails start-window
rows at 0.9944-0.9946 vs the 0.995 bar with `best_alt == pcc` (pure
gaussian-weight quantization noise, no flips) — the OPT-012 diagnostic that
motivated the real-weight controls above (logs/pytest_long_202k_moe.log).

## Performance accounting (moe decode, ctx 1024, batch 1, same run)

- Theoretical roofline: 40.2 MB moved per step at the deployment dtypes
  (attention bf4 12.5 + shared expert bf4 5.3 + top-4 experts bf4 21.2 +
  router fp32 0.5 + bf8 cache read 0.6 MB) / 512 GB/s p150 DRAM = **78.4 us**.
- Device time (tt-perf-report window): **333.7 us/step** — matmul rows
  172.9 us (dominant rows at 33-61% DRAM utilization: M=1-tile decode
  matmuls are launch/latency-bound on this arch; the sparse expert rows
  reach ~260 GB/s), non-matmul device ops ~161 us (SDPA 22.0, norms 13.6,
  binary eltwise 22.8, FillPad 18.0 — sparse_matmul's internal output
  zero-fill — layout/slice/transpose glue ~50, router/topk chain ~15).
- End-to-end: **491 us/token** = device 333.7 + ~157 us dispatch/replay gap
  over 67.9 ops (~2.3 us/op on Blackhole traced replay; the tracy gap metric
  reads 208 us including profiling overhead).

Named limitations feeding the ttnn backlog: per-op traced-replay dispatch
cost dominates the residual gap (fewer/larger ops is the only in-scope
lever; every packing/fusion/composite-op candidate for further reduction
was measured or has a recorded blocker - the ops/step went 61.2 -> 67.9
because the added reshard/slice glue belongs to measured winning
candidates);
sparse_matmul pays a full-output zero-fill before every call (~18 us/step
here, known from the fused stage, op-internal); sparse_matmul osw>1
multi-group corruption (bug above) blocks a faster union/prefill subblock
shape; TopK (k=4 runs as k=32 + slices) and the single-core embedding-pick
chain have ~15 us of latency floor. `perf_summary.json` records the
reconciliation for this decoder-layer workload.

## Contract preservation

- Public API, prefill/decode semantics, paged-cache geometry/ops/indexing,
  trace-capturability, determinism (prefill + traced decode bitwise) — all
  preserved and tested (same suites as the fused stage plus new coverage).
- decode_forward returns the residual in the width-sharded L1 config (the
  layer-to-layer contract for stacking); `to_torch` and re-feeding both work,
  and the input is resharded automatically if it arrives interleaved.
- Non-aligned logical lengths remain first-class (S = 17/65/509/1057/3000/
  202751 tested); padding/chunking stays internal.
- Batch: decode up to 32 users tested (union path, tuned osw=1 configs).
- KV cache: bf16 and bf8 both supported; dtype is an `allocate_kv_cache`
  parameter. Cache geometry/ops unchanged.
- Weight footprint: decode DRAM-sharded copies add ~0.55 GiB over 47 layers
  at the final dtypes (per moe layer: bf4 attention copies 9.9 MB + bf4
  shared_down 1.8 MB; dense layer adds its bf8 down copy) and the fused
  stage's prefill `wq_b_heads` copy remains; full-model projection at the
  final policy is ~17.5 GiB weights (including the copies) + 5.8 GiB bf8
  latent cache (or 10.9 GiB bf16) — fits 32 GiB at batch 1 with margin
  (details in `doc/context_contract.json`, updated by this stage).
- `doc/context_contract.json` gains an `optimized_decoder` section:
  `capability_reduction: none`, supported context 202752 at the same bars.

## Runtime fallback audit

`test_runtime_no_host_fallback`: `from_torch` / `to_torch` / `as_tensor`
tripwired during optimized prefill and decode — never fire; torch imports are
static-audited to setup only. Remaining measured-path layout ops, all
deliberate and measured: the decode q head split keeps one
untilize/reshape/tilize (the layout-free alternative, batched wq_b, is 3-4x
slower — rejected table); the prefill `slice_write` chunk accumulator
composite (inherited fused win); sharded<->interleaved hops at DRAM-sharded
matmul boundaries (0.6-1.1 us each, part of the measured winning candidates).

## Watcher

`TT_METAL_WATCHER=2` on the final code (decode PCC both kinds, moe-512
prefill, cache content, bf8-cache arms, dense traced decode, traced decode +
96-replay stress): 9 passed, no watcher exceptions/asserts/sanitize errors
(`logs/watcher/generated/watcher/watcher.log.gz`, pytest log
`logs/pytest_opt_watcher.log`). Watcher and profiler runs kept separate.

## Artifacts

- `perf_wallclock_*.json` — before/after wall clock (8 windows)
- `tracy/{moe,dense}/{prefill,decode}_{fusedbase,opt}_perf_report.{txt,csv,console.log}`
  + stacked CSV/PNG — tt-perf-report per window on the final code
- `long_context_{moe,dense}_attnbf8_mlpbf8.json` +
  `long_context_aligned_202752_attnbf8_mlpbf8.json` (synthetic comparability
  arm), `long_context_{moe,dense}_real_expertbf4_cachebf8.json` (deployment
  arm), `..._real_expertbf4_cachebf8_attnbf8_mlpbf8.json` and
  `..._real_expertbf4.json` (decomposition arms),
  `long_context_dense_real_expertbf4_cachebf8_mlpbf4.json` (rejected
  bf4-dense-MLP arm), `long_context_{moe,dense}_fusedcontrol_*.json` (fused
  real-weight controls; copied from the fused-decoder ladder outputs, their
  provenance is the paired logs), `*_legacy.json` (pre-MLP-policy
  decomposition arms, provenance via logs/work_log); every
  ladder-generated JSON from the final code carries an `arm` provenance block
- `logs/pytest_opt_{synth,real,perf_wall,watcher}.log`,
  `logs/pytest_long_*.log` (incl. fused real-weight controls), `logs/tracy_perf_run.log`
- `perf_summary.json` — roofline/device/e2e reconciliation
- `probe/optimize_perf_probe.py` — rerunnable isolated-op sweeps (all result
  tables recorded in `work_log.md`)
