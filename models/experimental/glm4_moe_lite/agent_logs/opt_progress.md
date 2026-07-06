# Optimization progress log (follow PLAN.md §3 loop)

Config under work: **batch 1, prefill, ISL 128, 2×4**. Metric = tt-perf-report category split
(minimize TM) + eliminate SLOW matmuls. No latency SLA.

## Baseline (iter 0) — minimal flag set
Flags: SKIP_DEFENSIVE_CLONES, FUSE_QKV_A, FUSE_SHARED_GATE_UP, DECODE_L1_ACT, EP_L1,
FUSE_MLP_MOE_REDUCE, SKIP_TYPECAST (+ WH: PREFILL_MATMUL_TUNED=0, CCL_NUM_LINKS=1).
Profile: `/tmp/glm_tracy_prefill2/reports/.../ops_perf_results_*.csv`.
Category split: **Compute 66.5% · TM 16.6% · Other 10.6% · DM 6.3%**.
Top: SparseMatmul 17.8%, Unary 14.9%, Matmul 14.9% (all SLOW), BinaryNg 10.0%, Concat 6.0%[TM],
AllGather 5.3%, ReduceScatter 4.9%, LayerNorm 4.7%, Slice 4.4%[TM,1581 ops].
All dense matmuls SLOW (5–33% util @ M=128).

## Iteration 1 — enable full production tuning flags
Hypothesis: several optimizations are OFF in iter0. Turning them on should cut TM (head
Slice/Concat) and matmul count:
- `FUSE_EXPERTS_GATE_UP=1` (fuse MoE gate+up → fewer expert matmuls + concat)
- `HEAD_PARALLEL_ATTN=1`, `HEAD_PARALLEL_KVB2=1`, `ATTN_DP=1` (default on, confirm)
- `SPARSE_MATMUL_PREFILL_TUNED=1`, `MOE_FAST_REMAP=1` (default on)
- `TP=1`
Change: re-profile prefill with the full sweep-script flag set (WH-adjusted). Measure category
split + SLOW count vs baseline. Correctness: these are the production flags that already ran e2e
in the ISL sweep, so low risk; still verify tokens after.
Status: DONE — **NO CHANGE**. Op graph byte-identical to baseline (SparseMatmul 368, Matmul 377,
Concat 187, Slice 1581 — all unchanged; TM still 16.6%). **Finding: these flags are already on by
default, so the TM overhead is STRUCTURAL — it will only move with code changes, not flag flips.**
SLOW matmuls unchanged (all dense matmuls 5–33% util @ M=128).

## Refined direction (for iter 2+)
Flag-tuning is exhausted for prefill. Real levers require code work + correctness verify + reprofile
(~10 min/cycle each). In priority order:
1. **TM Slice/Concat (structural, ~10% combined)** — the 1581 Slice + 187 Concat are per-layer
   head/KV/expert splitting in `attention_decode.py` / `decoder_layer_tt.py` / `moe_tt.py`. Each is
   tiny (~0.8 µs) → launch-overhead-bound, not data-bound. Win = fuse/eliminate via native-layout
   keeping or fused head ops. Needs code changes to the fused-op paths (risky; verify PCC each time).
2. **SLOW dense matmuls @ M=128** — under-utilized because M=128 is only 4 tiles (latency-bound).
   Program-config tuning (`sweeps/run_matmul_sweep.py`) can raise util somewhat, but the ceiling is
   low at M=128; bigger wins come at decode-batch-32 / larger prefill chunk where M is larger.
3. **Eltwise Unary 14.9% + BinaryNg 10.0% (25% combined!)** — SiLU/gating/residual. Candidate for
   fusing activations into the preceding matmul (fused_activation) to cut op count.

## Iteration 2 — adaptive `prefill_pcm` (kill needless MoE chunking at small ISL)  ✅ SHIPPED
Root cause found: `moe_sparse_experts_forward_tt` capped each sparse_matmul call to
`sparsity_block_size(32) * prefill_pcm` tokens, with `prefill_pcm` hard-defaulting to **1**.
At ISL-128 that forced **4 chunks/layer** — each chunk = per-chunk hidden/idx/weight/sparsity
**slices** + an output **concat** + 2 extra sparse_matmul launches. Pure launch + TM overhead;
chunking only exists to bound L1/DRAM on *long* prefills (MoE is token-wise → PCC-neutral).

Change (`moe_tt.py`): when `GLM4_MOE_LITE_MOE_SPARSE_PREFILL_PCM` is unset, compute `prefill_pcm`
adaptively = `min(blocks_needed, safe_blocks)` where `safe_blocks = SAFE_TOKENS(512)/block`. So a
prefill that fits the safe ceiling runs in **1 chunk**; long prefills still chunk. Added a cheap
`GLM4_MOE_LITE_MOE_CHUNK_DEBUG=1` decision print (no tensor calls; the old `MOE_SPARSE_DEBUG` path
crashes on `get_device_tensors`).

**Verified on 2×4 (batch-1, ISL-128), `ab_prefill_pcm_pcc.py`:**
- chunk decision OLD `prefill_pcm=1 → n_chunks=4` (×46 layers, + 184 recursive 32-tok sub-calls);
  NEW `prefill_pcm=4 → n_chunks=1` (×46), recursive sub-calls **gone**.
- **A/B logits PCC = 1.0, max_abs_diff = 0.0, argmax 551==551 → bit-identical.** Accuracy-neutral.

Win: per MoE layer 4→1 sparse_matmul chunks ⇒ ~¾ of the chunk-entry Slice/Concat TM ops and the
extra sparse_matmul launches removed at ISL≤512. Structural op-count reduction, zero PCC cost.

**MEASURED (ab_prefill_pcm_pcc.py, --time-iters 5, batch-1 ISL-128 eager, 2×4):**
- OLD (4-chunk) prefill median **492.3 ms** (min 489.1) → NEW (1-chunk) median **337.3 ms** (min 334.8)
- **delta −155 ms → 1.46× (+46% faster prefill)**, PCC 1.0 / argmax match (bit-identical).
- Chunk decision verified every iter: OLD n_chunks=4, NEW n_chunks=1 (×46 layers).

**tt-perf-report stacked report CONFIRMS the reduction** (see
`agent_logs/tt_perf_report_stacked_prefill_b1_isl128.txt`), baseline → post-opt:
- SparseMatmul **368 → 92** ops (46 layers × 2 = 1 chunk/layer), Slice **1581 → 569**, Concat 187 → 141.
- **TM share 16.6% → 9.6%** of device time. This is the single biggest TM win found so far and
  directly serves the "minimize TM" goal. Next TM targets from this report: Slice (569, 3.86%),
  ReduceScatter (DM 10%), Transpose/Concat/FillPad. Next SLOW-matmul target: the 377 dense Matmuls
  (mean FLOPs 33%) + lm_head 32×2048×38720 (SLOW).

Tooling added this iter (reusable by the autonomous agent):
- `scripts/ab_prefill_pcm_pcc.py` — full-depth A/B logits PCC (NEW vs OLD via env toggle) on 2×4.
- `agent_logs/verify_iter2.sh` — one-shot: chunk-decision debug + A/B PCC gate.
- `agent_logs/reprofile_iter2.sh` — tracy prefill reprofile → ops_perf_results CSV.

## Iteration 2b — prefill_pcm win across ALL batches (1/8/16/32) ✅
A/B PCC + timing at ISL-128 (`agent_logs/verify_multibatch.sh` + `/tmp/ab_b8b16_bp0.sh`):

| batch | flags | prefill OLD (4-chunk) → NEW (1-chunk) | speedup | PCC |
|------:|-------|---------------------------------------|--------:|----:|
| 1  | BATCHED_PREFILL=1, bf16 |   492.3 →   337.3 ms | **1.46×** | 1.0 |
| 8  | BATCHED_PREFILL=0, bf16 |  4095.3 →  2708.9 ms | **1.51×** | 1.0 |
| 16 | BATCHED_PREFILL=0, bf16 |  8163.0 →  5862.9 ms | **1.39×** | 1.0 |
| 32 | BATCHED_PREFILL=0, bf8  | 15695.3 → 11109.4 ms | **1.41×** | 1.0 |

All bit-identical (argmax 551 match every batch). **The prefill_pcm optimization is a
~1.4–1.5× prefill win across the entire batch matrix, zero accuracy cost.**

**Gotcha found (now an invariant): batch>1 prefill MUST use `GLM4_MOE_LITE_BATCHED_PREFILL=0`.**
My first batch-8/16 attempt used `BATCHED_PREFILL=1` and crashed with an L1 circular-buffer clash
(`program.cpp:1549`) — this reproduces with `PREFILL_PCM=1` too, so it is the batched-prefill path
being unsupported at batch>1, NOT the prefill_pcm change. batch 32 already used `=0`; using `=0` for
8/16 makes them run cleanly (table above). Added to PLAN.md invariants.

## Iteration 2c — across ISLs + SAFE_TOKENS ceiling correction
A/B (batch 1) at more ISLs:
- **ISL 256**: PCC=1.0, argmax match, prefill **650.7 → 396.6 ms = 1.64×**, 1 chunk (prefill_pcm=8).
- **ISL 512**: the initial `SAFE_TOKENS=512` default put all 512 tokens in ONE sparse call
  (per_core_M=16) → **L1 OOM** (`circular buffers grow to 2.37 MB > 1.5 MB`, program.cpp:1492).
  MEASURED safe ceiling is **256 tokens/call (per_core_M=8 ≈ 1.18 MB)**.

**Fix (committed): lower default `GLM4_MOE_LITE_MOE_SPARSE_SAFE_TOKENS` 512 → 256.** Now MoE
sparse calls never exceed 256 tokens; ISL≤256 runs in 1 chunk, larger ISLs chunk in 256-token
calls (e.g. ISL 512 → 2 chunks vs OLD 16) — win kept, no MoE OOM. This is the kind of bug the
"across all ISLs" testing is meant to catch (the 512 default would have OOM'd real ISL 384–511).

Separate note (NOT caused by prefill_pcm): running `runner.prefill` directly at **ISL 512** in the
A/B harness OOMs on a non-MoE op *before* the MoE even runs (crash precedes any chunk-decision
print, and reproduces on the OLD `pcm=1` path). The production sweep harness runs ISL 512 fine
(see `experiments/t3k_isl_sweep_b1`), so this is harness/flag-specific. **Worklist item P2:**
isolate the ISL≥512 direct-prefill OOM op (likely attention/dense-MLP CB at longer seq).

## Ground-truth PCC gate (TT vs HF) + cached HF reference
Added `scripts/pcc_vs_hf.py` (+ `agent_logs/pcc_vs_hf.sh`): computes the HF last-token prefill
logits ONCE on CPU (~59 GB model) and caches them to
`experiments/hf_ref/hf_prefill_lasttok_corpus_isl<N>.pt` (bf16, ~310 KB); later runs load the
cache and just PCC the TT model — instant. Real ground truth, not A/B.

**First result (ISL-128, PRODUCTION flags — BFP4 experts + LoFi):**
`PCC vs HF = 0.653`, **argmax token MATCHES (220==220)**. The matching top-token with a low
full-vocab PCC is the signature of *quantization noise*, not a broken forward: the shipping config
runs 4-bit experts + LoFi fidelity across 46 MoE layers. The official ≥0.97 pipeline test uses
bf16 experts + fp32 acc + TP=0 to isolate implementation from quantization.

Implications:
- **A/B-vs-shipping** (ab_prefill_pcm_pcc.py) stays the correctness gate for *changes* (detects
  regressions precisely; the prefill_pcm change scored 1.0 there).
- **HF-PCC** (this script) measures *deployed accuracy vs ground truth*. Under production quant it's
  ~0.65 (top-token correct). TODO to confirm quant-vs-bug: rerun under conservative env
  (`GLM4_MOE_LITE_EXPERTS_TT_DTYPE=bf16 GLM4_MOE_LITE_MOE_FP32_ACC=1 TP=0 ATTN_DP=0`) — expect ~0.97
  (one-time weight-cache regen). If production distribution fidelity matters, consider bf8 experts.

## PCC investigation (user priority: "no optimization until PCC passing")
Root-caused why WH prefill-logits PCC vs HF was 0.65. Full ladder (ISL-128, cached HF ref via
`scripts/pcc_vs_hf.py`; per-vocab-shard probe shows the error is UNIFORM, not an lm_head assembly bug):

| config | PCC | note |
|---|---|---|
| TP=1 / bf4 (production) | 0.653 | baseline |
| TP=1 / bf8 (clean) | 0.656 | dtype barely matters |
| TP=1 / lm_head=bf16 | 0.662 | lm_head not the cause |
| **TP=0 / bf4** | **0.935** | TP=0 fixes it |
| **TP=0 / bf8 + MOE_FP32_ACC** | **0.954** | best WH-fitting |
| TP=0 / bf16 experts (the 0.97 config) | OOM | Blackhole-only |
| TP=1 / ATTN_DP=0 | 0.652 | **ATTN_DP not the cause** |

**ROOT CAUSE: the TP=1 (tensor-parallel) path is the accuracy bug** — TP=0 gives 0.935, TP=1 gives
0.65 at identical dtype, and disabling ATTN_DP doesn't help. The error is uniform across vocab and
compounds over layers → it's in the TP-sharded transformer body (sharded q_b/kv_b/w_o + shared-MLP
matmuls and their AllGather/ReduceScatter combination), NOT quantization, NOT lm_head, NOT ATTN_DP.

Also: precision knobs `MOE_SPARSE_FP32_ACC` / `MOE_SPARSE_FIDELITY=hifi4` / `MOE_SPARSE_APPROX=0`
REGRESS PCC (0.954→0.54) → those paths are buggy too.

**Deliverables:** WH PCC test `tests/pipeline_tests/test_text_prefill_logits_wh.py` (gate 0.95,
PASSES at 0.9516, TP=0/bf8) + `apply_wh_correctness_env`. bf16 experts OOM WH (~7GB/chip) so 0.97
is Blackhole-only; 0.95 is the WH ceiling.

**Next (P0 correctness): localize the TP=1 bug** — run single-layer PCC (test_tt_decoder_layer0 =
attention+dense MLP; test_tt_moe_layer1 = MoE) at TP=1 vs torch to find which sharded op diverges,
then fix the sharded-matmul partition or the CCL combine. Production needs TP=1 for perf, so this
gates all optimization work.

### TP=1 bug LOCALIZED (test_tt_decoder_layer0_prefill_update_cache_optional.py, mesh 1x4)
Ran the built-in TP-vs-no-TP PCC checks (which compare TP=1 output to replicated no-TP, target 0.999):
- **`test_..._tp_matches_no_tp` (layer 0 = attention + dense MLP): PASS, PCC ≥ 0.999.** → attention +
  dense-MLP TP sharding is CORRECT.
- **`test_layers_through_1_moe` (through first MoE layer): documented "first known TP divergence
  ~PCC 0.95".** `test_incremental_per_layer_tp_pcc`: PASS (asserts the known divergence).
- Conclusion: **the TP=1 accuracy bug is in the MoE-layer TP path** (shared-expert gate/up/down TP
  sharding and/or the `_moe_all_reduce_across_mesh` combine), ~0.95 per MoE layer, compounding over
  46 MoE layers → 0.65 full-model. It is a KNOWN, pre-existing divergence (documented in the test).
- Fixed a real hazard while here: those TP localizer tests hardcoded `CCL_NUM_LINKS=2` (deadlocks on
  T3K, 1 link/axis) — changed to respect an outer override, default 1, so they run on T3K.
- **FIX SCOPE (not done): correct the MoE-layer TP sharded compute / all-reduce so TP=1 matches
  no-TP to ~0.999 per layer; then production (TP=1) should reach the ~0.95 HF ceiling.**

### ROOT CAUSE (2x4 TP=1): invalid FUSE_MLP_MOE_REDUCE on 2D meshes
`decoder_layer_tt.py` MoE path: with `FUSE_MLP_MOE_REDUCE=1` (default) + tp_enabled,
`_skip_shared_reduce=True` (L768). Then:
- shared expert all_reduce is SKIPPED (L1331) — shared_out left TP-column-partial;
- routed experts run with `skip_final_reduce=True` (L1392) — their `_moe_all_reduce_across_mesh`
  (which reduces over BOTH mesh axes) is skipped, leaving routed_out EP-partial;
- `mlp_out = shared + routed`, then ONE `all_reduce(cluster_axis=tp_axis)` (L1405-1412).

The bug: that single reduce covers only `tp_axis` (columns, size 4). But:
- routed (EP) is sharded across ALL 8 chips → needs reduction over BOTH axes; the **DP-row axis
  (size 2) sum is dropped** → ~half the routed-expert contribution missing → full-model PCC 0.65.
- shared is column-partial AND replicated across the DP rows → it must NOT be reduced over rows
  (would double-count). So shared and routed **cannot share one all_reduce on a 2D (DP>1) mesh**.

Evidence chain: TP=0 (no fusion, replicated) = 0.935 ✓; 1x4 (DP=1, single axis covers both) ~ok;
2x4 (DP=2) = 0.65 ✗. Confirmed the fusion is the site; `FUSE_MLP_MOE_REDUCE=0` on 2x4 CRASHES
(`all_gather num_devices=1`), so the non-fused path also has a 2x4/DP bug (size-1 axis).

**Fix design (delicate, distributed):** the shared+routed reduce fusion is only correct on 1D
meshes. On 2D meshes with DP>1: reduce shared over `tp_axis` only, reduce routed over the FULL mesh
(both axes), then add — i.e. gate `_skip_shared_reduce` on `dp_axis_size == 1`, AND fix the
non-fused routed path's size-1-axis crash so it runs on 2x4. Verify with the per-layer TP test
(target 0.999/layer) then full-model pcc_vs_hf (expect ~0.95 at TP=1).

## Sweep verified on WH LB with the accuracy (PCC>=0.95) config
Made `run_sweep_isl_batch.py` knobs env-overridable (TP/ATTN_DP/EXPERTS_TT_DTYPE/MOE_FP32_ACC
were hard-set to the TP=1 perf config that scores 0.65 PCC). Ran the sweep with the 0.95-passing
knobs (TP=0, bf8 experts, MOE_FP32_ACC=1), batch 1:

| ISL | prefill_s | per-user tok/s | TTFT ms | decode ms | status |
|----:|----------:|---------------:|--------:|----------:|--------|
| 128 | 0.3 | 16.5 | 404  | 60.6 | ok |
| 512 | 1.6 | 16.2 | 1651 | 61.6 | ok |
| 2048| —   | —    | —    | —    | L1 OOM (program.cpp:1492) |

Tradeoff vs production (TP=1 bf4 ~20.8 tok/s but PCC 0.65): the accuracy config is ~20% slower
(TP=0 = no tensor-parallel) and memory-limited at long ISL. Confirms the value of fixing the TP=1
bug (get both ~20.8 tok/s AND 0.95 PCC). Sweep output artifacts NOT committed (kept local);
numbers recorded here.

## Reality note
Each remaining iteration is deep fused-op code surgery + 10-min reprofile + PCC verify. This is the
long autonomous grind the PLAN is built for; it is not flag-flipping. Batches 8/16/32 repeat this
loop (decode M grows with batch → matmul util improves, so SLOW-matmul pressure eases at higher batch;
TM structural overhead persists).

## WH 1x8 optimization loop (batch-1, TP=1 correct config)

After the TP=1 fix (1x8 mesh) + batch-1 baseline landed, profiled the 1x8 baseline with
Tracy (agent_logs/profile_b1_wh_1x8.sh). Device-time breakdown (eager, prefill-dominated):
SparseMatmul 36% (MoE experts, LoFi+tuned), ReduceScatter 28% + AllGather 9.5% (MoE/attn
all-reduce, ~3x their 2x4 share — 8-device linear all-reduce is O(N) latency), Matmul 4.8%.

Committed wins (each: tracy -> reason -> ISL perf test -> PCC vs HF 128..8192 -> commit):
  1. Ring all-reduce (2b7a9fb2ad0): Topology.Linear -> Ring on the 8-device line.
     decode 93.5 -> 88.4 ms/tok (-5.5%), prefill neutral, PCC unchanged. num_links=1 (no
     deadlock). Scoped to WH via apply_wh_tp1_env; code default stays Linear (BH untouched).
  2. WH HiFi3 for fp32-accum MLP/router matmuls (6219af33add): HiFi4+fp32-accum hits a WH HW
     bug (slower AND less accurate; ttnn warns to prefer HiFi3). prefill 7.19 -> 6.79s (-5.6%),
     PCC held/improved (4096: 0.9304 -> 0.9330). is_wormhole_b0()-gated; BH keeps HiFi4.

Dead-ends (reasoned, NOT committed):
  - prefill chunk 128 -> 256: SLOWER. Pushes MoE sparse matmul into num_blocks>1 (forced to
    DRAM, 2-4x slower per moe_tt.py). Keep chunk=128.
  - CCL_NUM_LINKS 1 -> 2 (with ring): DEADLOCK (timeout). Confirms T3K's 1-CCL-link/axis HW
    limit is physical, not topology-dependent.

Remaining bottlenecks are HW/algorithm-limited:
  - SparseMatmul (36%): already LoFi + tuned + fused gate/up; larger M regresses it. No knob.
  - Collectives (37%): ring took the decode latency win; prefill collectives are bandwidth-
    bound and can't add links (1-link HW). Only lever is bf8 reduce, which is too risky — ISL
    4096 PCC is already 0.933 (gate 0.93), so halving reduce precision would likely fail.
  - Attention runs head-parallel OFF on 1x8 (20 heads not divisible by tp=8) -> replicated.
    Enabling it needs head-padding 20->24 + new padded weight variants: ~3-5% upside, high
    correctness risk, poor ROI on a hang-prone device. Deferred.

## 2x-prefill goal — diagnostic drill complete, traced-prefill is the lever

Tracy + tt-perf-report on the WH 1x8 batch-1 prefill (ISL=2048):
  * Prefill wall (~6.8s warm) is ~53x the device-kernel time (~127ms/dev). tt-perf-report
    Bound column is BLANK on every op with Op-to-Op Gap >> Device Time -> prefill is
    HOST-DISPATCH bound, not compute bound.
  * Device-kernel stack (the compute floor): Compute 53.6% (SparseMatmul/MoE 38.5%),
    DM/collective 34.3% (ReduceScatter 25.7% + AllGather 7.5%), TM 8.4%.
  * Matmuls all marked SLOW at M=128 (FLOP util 16-51%) — small-M chunks underutilize cores.

Standalone probe (/tmp/trace_probe.py, pure ttnn 1x8, 47-layer matmul+all_reduce stack):
  EAGER 31.6 ms/pass vs TRACED 9.9 ms/pass = 3.21x. Confirms tracing collapses dispatch
  and yields >2x on the actual prefill TIME metric. This is THE lever for the goal.

Dead-ends (verified, not committed):
  * a2a MoE dispatch (GLM4_MOE_LITE_MOE_SPARSE_DISPATCH_IMPL=a2a) DEADLOCKS on 1x8
    (0.5% CPU, blocked at chunked prefill) — same as 2x4. ttnn all_to_all_dispatch not
    viable on T3K as configured.
  * DeepSeek prefill MoE micro-ops (ttnn.experimental.deepseek_prefill.{dispatch,combine,
    unified_routed_expert_moe,post_combine_reduce}, deepseek_moe_reduce_scatter) are the
    real kernel-time lever (skip-non-local reduce ~75% MoE compute + reduce_scatter-to-
    sharded), BUT are built for 2D dispatch-group meshes (2x4/8x4/8x1) — deepseek_v3_d_p
    perf tests don't target 1x8. Adopting them means re-architecting GLM MoE around
    dispatch/combine AND moving back to a 2D mesh (reopens the 1x8-vs-2x4 TP decision).
    Deferred (goal path #2).

Traced-prefill implementation scope (goal path #1, the chosen lever):
  * chunked_flash_mla_prefill takes chunk_start_idx as a per-chunk int -> each chunk is a
    distinct program, so NOT one shape-stable trace (unlike decode). Viable form is a
    WHOLE-prefill trace: capture all chunks' programs once (warmup), execute_trace on the
    timed/subsequent same-length call.
  * Requires hoisting per-chunk HOST input-prep out of the traced region into persistent
    device tensors: run_tt_embedding does from_torch internally (host), and each chunk does
    from_torch for chunk_page_table. Pre-stage tokens + page_tables as persistent device
    buffers (mirror run_tt_decode_embedding's tokens_tt pattern), trace reads from them.
  * Trace-region limit: full-prefill trace grows with num_chunks; very long ISL (131072 =
    1024 chunks) may not fit -> fall back to eager for lengths beyond a cap (documented).
  * Expected: >2x on prefill time for all lengths that fit the trace (benchmark + typical
    serving). PCC unchanged (same ops, just replayed device-side).
