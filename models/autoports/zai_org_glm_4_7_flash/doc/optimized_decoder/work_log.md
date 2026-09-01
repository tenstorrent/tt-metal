# GLM-4.7-Flash optimized decoder — work log

Stage: optimized-decoder (single Blackhole p150-class chip, device 0, 1x1 mesh).
Start state: fused decoder complete (commit e1eb9a45905, stage evidence in
`doc/fused_decoder/`). Deliverable: `tt/optimized_decoder.py` + tests + docs.

Hardware: p300c board device 0; compute grid 11x10 (110 cores), DRAM grid 8x1
(8 banks), arch blackhole. `tt-smi -ls` healthy at stage start (4 boards).
Device smoke: open/close device 0 OK.

Deployment dtype contract (from goal + doc/probe/README.md): routed experts
MUST be bfloat4_b from the full-model stage onward; this stage optimizes with
bf4 experts as the default arm and keeps the bf8 arm for comparison.

## Operation-topology audit (measured fused path, before optimization)

Source: `doc/fused_decoder/tracy/{moe,dense}/decode_fused_perf_report.csv`
(traced decode, batch 1, ctx 1024, 32-replay signpost window; per-op means),
wall clock from `doc/fused_decoder/perf_wallclock_fused_decode_*.json`.
Fused headline: moe 1.035 ms/tok wall, 787 us/step device, ~61 ops/step;
dense 0.969 ms/tok wall, 890 us/step device. Overall modeled DRAM roofline
in the fused report: 15.9% (81 GB/s) — the decoder is heavily config- and
dtype-bound, not roofline-bound.

MoE decode step (dominant rows, mean device us, effective GB/s at current dtype):

| role | shape M×K×N | dtype now | us | GB/s | candidate action |
|---|---|---|---|---|---|
| w_uv absorbed | b=20 ×32×512×256 | bf16 | 121.7 | 43 | program config (deepseek mla1d batched-DRAM-sharded idiom) + bf8 |
| sparse gate_up (top-4) | 4E ×32×2048×3072 | bf8 | 118.9 | 225 | bf4 + geometry sweep (cores/in0_block_w/subblocks), L1 out |
| wo | 32×5120×2048 | bf16 | 91.7 | 229 | bf8 + DRAM-sharded matmul |
| sparse down (top-4) | 4E ×32×1536×2048 | bf8 | 71.7 | 186 | bf4 + geometry sweep, L1 out |
| shared gate+up | 2× 32×2048×1536 | bf16 | 67.8 | 186 | bf8 + DRAM-sharded |
| w_uk absorbed | b=20 ×32×192×512 | bf16 | 53.6 | 73 | program config + bf8 |
| 4× rms_norm | [32,2048]/[32,768]/[32,512] | bf16 | ~22.4 each (1 core!) | - | sharded norm program config + L1-sharded residual |
| wqkv_a | 32×2048×1344 | bf16 | 33.1 | 167 | bf8 + DRAM-sharded |
| shared down | 32×1536×2048 | bf16 | 28.9 | 218 | bf8 + DRAM-sharded |
| SDPA decode | ctx 1024, bf16 latent cache | - | 28.8 | - | bf8 cache candidate (OPT-002 mandatory), config sweep |
| router | 32×2048×64 | fp32 w | 25.5 | 20 | program config/grid; keep fp32 selection semantics |
| wq_b | 32×768×5120 | bf16 | 21.7 | 362 | bf8 + DRAM-sharded |
| topk+gather+score chain | [32,64] fp32/bf16 | - | ~45 | - | shrink op chain; keep selection semantics |
| eltwise/slice/layout glue | many small | - | ~120 | - | width-sharded L1 residual/intermediates; fewer conversions |

Dense decode step: MLP gate+up 2×106.5 us (394 GB/s, bf16), down 180.9 us
(232 GB/s), plus the same attention rows as moe. Dense MLP → bf8 +
DRAM-sharded (+ BFP4 trial mandated by skill when MLP dominates: here MLP is
~500/890 us of the dense step).

Repeated same-input matmul groups already packed at the fused stage: wqkv_a
(q_a+kv_a), packed routed gate_up. Shared-expert gate+up kept separate at the
fused stage (no per-group mcast to save, silu fold would be lost); revisit
under DRAM-sharded configs where the input read is amortized differently.

Dispatch gap: moe ~248 us/step over ~61 ops (~4 us/op on this dispatch path);
dense ~109 us/step over 44 ops. Op-count reduction and bigger ops are part of
the target.

Prefill (S=2048 warmed wall): moe 210 ms (gate_up sparse stage 68.1 ms/chunk
dominates), dense 15.3 ms. Prefill actions: 2D mcast program configs for the
big matmuls, bf4 expert trial (compute-bound at LoFi could ~2x the expert
matmul throughput), keep fp32-acc HiFi4 flash prefill (long-context drift
evidence from the functional stage).

## Precision/fidelity policy (named, per skill)

Starting policy for the optimized arm (tune one group at a time):
- activations/residual: bf16; norms bf16 (HiFi4 norm kernel config as now)
- attention weights (wqkv_a, wq_b, w_uk, w_uv, wo): bfp8 + fidelity sweep (HiFi2 vs LoFi)
- shared-expert + dense-MLP weights: bfp8 (+ mandatory bfp4 trial; MLP dominates dense decode)
- routed experts: bfp4 + LoFi (deployment contract; bf8 arm kept for comparison)
- router gate weight: fp32 (selection semantics), config-tuned
- latent KV cache: bfp8 candidate vs bf16 (OPT-002; full-model projection in
  context_contract assumes bf8 cache fits 202k ctx)
- flash prefill: fp32-acc HiFi4 (unchanged; drift evidence)
- flash decode: HiFi4 no-fp32acc (unchanged unless swept faster + PCC holds)

## Dtype-only arms (fused decoder, traced decode moe ctx1024, synth weights)

`tests/dev_optimize.py` (same methodology as test_fused_perf: trace, 3 warm, 32 timed):

| arm | decode ms/tok | pcc prefill509 / decode |
|---|---|---|
| A: bf16 weights + bf8 experts (baseline re-measure) | 1.0340 | 0.999378 / 0.99997x |
| B: bf8 weights + bf8 experts (dtype only, same HiFi4-fp32acc cks) | 1.0297 | 0.999501 / 0.99997x |

Conclusion: decode is config/dispatch-bound, not weight-BW-bound; program
configs and layout are the lever. (bf8/bf4 still matter for prefill compute
and for the roofline once configs are fixed.)

## Isolated-op probe results (probe/optimize_perf_probe.py, traced, us/op)

Absorbed matmuls (the two worst rows):

| role | candidate | us | pcc vs torch |
|---|---|---|---|
| w_uk b20 32x192x512 | default bf16 hifi4fp32 (current) | 54.1 | 0.999998 |
| w_uk | reuse 5x4 bf8 hifi2 (interleaved in0) | 16.5 | 0.999945 |
| w_uk | batched-dram pad24 bf8 hifi2 / lofi | 17.4 / 13.8 | 0.999945 / 0.999875 |
| w_uv b20 32x512x256 | default bf16 hifi4fp32 (current) | 122.1 | 0.999996 |
| w_uv | reuse 5x4 bf8 hifi2 (interleaved in0) | 21.1 | 0.999908 |
| w_uv | batched-dram pad24 bf8 hifi2 / lofi | 20.9 / 16.0 | 0.999906 / 0.999855 |

batched-dram at bf16 fails (static CB clash with L1 buffers, program 17) —
bf8 fits. batched-dram needs heads padded 20->24 (8 banks), L1 height-sharded
in0 on the optimal-worker cores; reuse 5x4 keeps the current interleaved
layout. Integration starts with reuse 5x4 bf8 hifi2; batched-dram is the
follow-up candidate (~8 us/step better isolated, but needs padded-head
plumbing whose cost must be measured in-layer).

RMS norms ([1,1,32,w] bf16, current = 1 core):

| width | default | sharded best | grid |
|---|---|---|---|
| 2048 | 34.9 | 6.7 | 16c (8x2) or 32c |
| 768 | 14.4 | 5.4 | 8c |
| 512 | 10.4 | 5.2 | 8c |

Router matmul 32x2048x64 (fp32 weight, fp32 out, hifi4 fp32acc kept):

| candidate | us | pcc |
|---|---|---|
| default (current) | 25.8 | 0.999999 |
| 1D mcast 2 cores, in0_block_w=32 | 8.5 | 0.999998 |

## Sparse matmul geometry sweep (probe, bf4/bf8, decode-shaped indexed mode)

Baseline (fused config): gate_up pcn3 bw8 osw1 bf8 hifi2+fp32acc 125.4 us,
down pcn2 bw6 osw1 78.8 us. Winners: gate_up 8x6 pcn2 bw8 osw2 bf4 58-62 us;
down 8x8 pcn1 bw16 osw1 bf4 L1 34.9 us (pcn2 bw24 osw2 dram 35.0 equal —
osw1/pcn1 kept). Best bf8 rows for comparison: gate_up 82.0, down 44.6 —
bf4 wins on perf and is the deployment dtype.

## TTNN op bug found: sparse_matmul out_subblock_w>1 corrupts multi-group outputs

In the NON-indexed sparsity-walk mode, out_subblock_w=2..3 (out_block_w=osw)
corrupts multi-group outputs: G=4-group prefill-shaped gate_up at bf4,
PCC vs torch: osw=1 -> 0.9939 (any pcn/bw: pcn3bw8, pcn2bw32, pcn3bw16);
osw=2/3 -> 0.82-0.87 (pcn2bw32osw2 0.870, pcn3bw32osw3 0.829, pcn2bw8osw2
0.822; corruption grows with K-block count). The indexed/gather mode is
immune: identical PCC 0.993634 for osw1 vs osw2 at the same config. Repro:
`probe/optimize_perf_probe.py` role_sparse + the CASE script recorded in this
log (sparse_matmul, [1,G,32,2048]x[1,64,2048,3072], sparsity all-ones,
nnz=G*E, MatmulMultiCoreReuseMultiCast1DProgramConfig(out_subblock_w=2,
out_block_w=2)). Consequence for this stage: indexed batch-1 decode keeps
osw=2 (verified immune + faster: 58 vs 107 us); union (batch>1) and prefill
paths pin osw=1. Candidate ttnn issue for the backlog.

## Prefill sparse geometry (osw=1, per 1024-token all-ones chunk)

| matmul | fused config | tuned | ms |
|---|---|---|---|
| gate_up | pcn3 bw8 (8x4) 61.4 | pcn3 bw32 (8x4) | 37.4 |
| down | pcn2 bw6 (8x4) 37.8 | pcn1 bw24 (8x8) | 18.3 |

LoFi vs hifi2+fp32acc: no speed change (mcast/BW-bound) -> prefill experts
keep hifi2+fp32acc. Warmed moe prefill S=2048: 195 -> 126 ms.

## Packed-vs-separate gate/up under tuned configs (OPT-010, probe)

- shared expert (2048->1536 x2): separate (silu fused in gate matmul) 26.75 us
  vs packed (1 matmul + 2 slices + silu-in-mul) 26.97 us -> keep separate.
- dense MLP (2048->10240 x2): separate 137.6 us vs packed 139.1 us -> keep
  separate. (Fused-stage analytic rejection now backed by measurement.)

## num_workers_per_dram_bank=2 (Blackhole knob) rejected with evidence

wo 38.7 vs 26.0 us; shared_down 15.9 vs 11.1; wqkv 16.1 vs 10.8; dense_gate
96.0 (nw2, legal) vs 71.2 (wide-1D winner; nw1 dram fails with CB clash at
bw8 program.cpp:1932). All four roles keep nw=1 / wide-1D.

## wq_b wider-shard candidate (OPT-004): 4 cores bw6 14.0 us vs 8 cores bw3
13.1 us -> keep 8c bw3 (recorded; K=768=24t limits 8-core in0_block_w to 3).

## q-path fold candidates rejected (probe role_qpath)

fold W_UK into wq_b per-head ([1,20,768,512] batched): MatmulMultiCoreReuse
with broadcast LHS is illegal (batch-dim TT_FATAL; in0-reuse requires 1D
non-mcast + all interleaved). Legal variants measured: bcast1d 373-470 us,
repeat+reuse 59.6-64.6, prerepeated reuse 53.4 — all lose to the current
split chain (wq_b 12.7 + w_uk 13.1 + glue ~30 us). Rejected with numbers.

## Output-path candidates rejected (probe role_wopath)

Batched per-head wo + fast_reduce_nc: 80.8 us (lofi) vs current
v-chain + DRAM-sharded wo ~56 us -> rejected. Host-folded w_uv@wo per head:
CB overflow (program.cpp:1875) at [1,20,512,2048] reuse config -> blocked.

## Router gather replaced by embedding-table pick (batch-1 indexed path)

ttnn.gather at [1,1,32,64] is a single-core ~37 us kernel (fp32/bf16/L1 all
equal). Replacement: typecast scores bf16 -> RM [64,1] table -> ttnn.embedding
at the topk ids -> [1,k,1,1] compact picks; normalization on the compact
picks. Probe: 18.3 us for the whole chain. Union path (batch>1) keeps the
dense scatter-mask math (no gather).

## bf8 latent cache adopted (OPT-002)

paged MLA flash decode at ctx1024: bf16 40.7 vs bf8 40.1 us isolated (k256+
fails CB, mc16 slower -> keep k_chunk=128, max_cores_per_head_batch=8).
End-to-end with bf8 cache: prefill kvpe typecast to cache dtype before
paged_fill_cache; decode paged_update_cache keeps bf16 input. Layer PCC
(synth, bf4 experts): prefill 0.99719 vs 0.99742 at bf16 cache; real-weight
deployment arm 0.99693/0.99473 agg (fused bf4/bf16-cache control:
0.99707/0.99489). Cache content vs exact linear ref at bf8: 0.999938.
bf8 cache halves full-model cache bytes (10.9 -> 5.8 GiB at 202k).

## OPT-007 bf4 attention weights: adopted (real-weight evidence)

Trial (attn_weight_dtype=bfloat4_b on wqkv_a_ds/wq_b_ds/wo_ds + absorbed
w_uk/w_uv; shared/dense MLP + prefill flat copies stay bf8):

| arm | prefill509 PCC | decode steps | traced decode |
|---|---|---|---|
| moe REAL bf4exp+bf8cache, attn bf8 | 0.99706 | 0.9976/0.9978/0.9975 | 0.5135 |
| moe REAL bf4exp+bf8cache, attn bf4 | 0.99574 | 0.9975/0.9977/0.9975 | 0.5000 |
| dense REAL, attn bf4 | 0.99997 | 0.99992+ | 0.4491 |
| dense synth, attn bf4 | 0.99987 | 0.99991+ | 0.4498 |
| moe synth bf8exp, attn bf4 | 0.99767 | 0.99971+ | 0.5397 |

Real-checkpoint decode PCC unchanged; synthetic-weight prefill drops more
(adversarial gaussian quantization, OPT-012) but stays above bars. The 202k
ladder at the final policy is the long-context decider (below).

## Same-session before/after (test_optimized_perf.py, 8 windows)

| window | fused baseline | optimized | delta |
|---|---|---|---|
| moe traced decode ctx1024 | 1.035 ms/tok | 0.513 ms/tok (bf8-attn) / 0.500 (bf4-attn) | -50/-52% |
| dense traced decode | 0.969 | 0.463 / 0.449 | -52/-54% |
| moe warmed prefill S=2048 | 210.3 ms | 125.9 ms | -40% |
| dense warmed prefill S=2048 | 15.2 ms | 11.1 ms | -27% |

tt-perf-report (tracy/, 8 signposted windows): moe decode DRAM roofline
15.9% -> 38-43%; dominant rows show `LoFi BF16 x BFP8` (now BFP4) proving the
policy reached the ops (OPT-013). Remaining moe decode step: 354 us device +
210 us dispatch gap over 68 ops; top rows: sparse gu 48.5, dn 31.5 (bf4,
~260 GB/s), wo 25.6, SDPA 22.0 (bf8 cache), shared gate/up 2x11.3,
w_uv 16.1, w_uk 13.1, wq_b 12.7, norms 4x~5.4, eltwise/glue rest.

## Correctness suites (final code)

- synth suite `pytest test_optimized_decoder.py -m "not real_weights"`:
  23 passed (incl. dense traced PCC + batch8/bf8-cache cross) (logs/pytest_opt_synth.log): prefill 17..3000 both kinds
  0.99933-0.99999, decode agg 0.99584 (pos-509 sub-ulp tie proven), dense
  0.99995, batch8 all >=0.9998, batch32 31/32 + 1 proven tie, traced replays
  0.9999, 96-replay stress + bitwise determinism, cache content 0.99996,
  bf8-cache arm, no-host-fallback tripwire, construction audit.
- real suite: 3 passed (logs/pytest_opt_real.log): moe 0.99924/0.99697 agg
  (pos-514 tie, same as fused), dense 0.99998/0.99994, deployment arm
  bf4+bf8cache 0.99693/0.99473 agg.

## Long-context arms and the anomaly ledger (bf4 attention / bf8 cache at 202k)

Comparability arm (synthetic weights, bf8 attention + bf8 experts + bf16
cache — the fused ladder's exact arms): **5 passed** at the 0.995 bar
(logs/pytest_long_synth_attnbf8.log). moe end window 0.994927 agg, 28/32 rows
+ 4 explained flips, 0 unexplained (fused: 0.994917, 28/32 + 4); dense
control 32/32 in all three windows (end agg 0.999701 vs fused 0.999704);
decode@202751 moe 0.999900 / dense 0.999951; aligned-202752 final window
28/32 + 4 explained. The layout/program-config/LoFi work preserves the 202k
contract exactly.

Deployment-policy arms then produced an anomaly that was fully controlled:

Observed anomaly: real-weight 202k end-window rows at 0.985-0.99 (below the
  0.99 deployment bar on 2-3 rows) with bf4 attention and/or bf8 cache;
  synthetic ladder at bf4 attention fails start-window rows at 0.9944-0.9946
  vs the 0.995 bar with best_alt == pcc (pure quantization noise, no flips).
Evidence: logs/pytest_long_202k_moe.log (synth bf4-attn),
  logs/pytest_long_real_deployment.log (bf4attn+bf8cache),
  logs/pytest_long_real_attnbf8_cachebf8.log,
  logs/pytest_long_real_attnbf4_cachebf16.log.
Affected path: prefill window rows at full 202k context (decode@202751
  passes every arm: 0.9977-0.9999).
Control or comparison: the FUSED decoder (bf16 weights, HiFi4, bf16 cache —
  the accepted previous stage) rerun on REAL weights at 202k:
  dense end window 0.993950 agg, 29/32 rows at 0.99, 3 unexplained
  (logs/pytest_long_real_fused_dense.log); moe (bf4 experts) end window
  0.987522 agg, 23/32 + 7 explained + 2 unexplained
  (logs/pytest_long_real_fused_moe.log).
Likely subsystem: none in the optimized decoder — the sub-0.99 end-window
  rows are a property of the real checkpoint at full context and of the
  flip-reconstruction machinery's ceiling under bf4-expert noise; they exist
  identically (slightly worse) in the fused baseline.
Investigation performed: 4-arm dtype decomposition on real weights (attn
  bf4/bf8 x cache bf8/bf16): dense end agg 0.994196 / 0.994045 / 0.994293 /
  fused-control 0.993950 — all within 0.0004, optimized >= control in every
  arm; moe deployment arm 0.988530 / 24 rows vs fused control 0.987522 / 23
  rows. Synthetic bf4-attn failure isolated to gaussian-weight bf4 block
  quantization (real-weight arms unaffected), the OPT-012 pattern.
Resolution: controlled. Final policy keeps bf4 attention + bf8 cache
  (real-weight evidence: equal-or-better than the accepted baseline at every
  context; decode faster). Synthetic suites/ladder run the bf8-attention +
  bf16-cache comparability arm at the historical 0.995 bar; the deployment
  arm is validated on real weights at the 0.99 deployment bar
  (long_context_{moe,dense}_real_deployment.json; deterministic rerun
  reproduced the rows bit-for-bit).

## Final policy and headline (same session, 8 windows)

attention bf4 (decode DS copies + absorbed w_uk/w_uv) / experts bf4 /
shared+dense MLP bf8 / prefill flat copies bf8 / router fp32 / cache bf8
deployment (bf16 supported) / LoFi decode / HiFi2+fp32acc prefill matmuls /
HiFi4+fp32acc norms + flash prefill.

| window | fusedbase | opt | delta |
|---|---|---|---|
| moe traced decode ctx1024 | 1.034 ms/tok | 0.500 | -51.6% |
| dense traced decode | 0.969 | 0.450 | -53.6% |
| moe prefill S=2048 | 210.2 ms | 126.7 | -39.7% |
| dense prefill S=2048 | 15.2 ms | 10.3 | -32.2% |
| moe decode device/step | 787.3 us (61.2 ops) | 340.7 us (67.9 ops) | -56.7% |

Performance accounting (moe decode): roofline 87.7 us (44.9 MB/step at
512 GB/s) / device 340.7 us / e2e 500 us; reconciliation and named
limitations in README.md + perf_summary.json.

## Commands (final evidence runs)

```bash
# correctness (final code)
pytest models/autoports/zai_org_glm_4_7_flash/tests/test_optimized_decoder.py -q -s -m "not real_weights"
pytest models/autoports/zai_org_glm_4_7_flash/tests/test_optimized_decoder.py -q -s -m "real_weights"
# long context: comparability arm + deployment arm + fused controls
GLM47_DECODER=optimized GLM47_ATTN_DTYPE=bf8 pytest models/autoports/zai_org_glm_4_7_flash/tests/test_long_context.py -q -s -m long
GLM47_DECODER=optimized GLM47_REAL_WEIGHTS=1 GLM47_EXPERT_DTYPE=bf4 GLM47_CACHE_DTYPE=bf8 GLM47_PCC_BAR=0.99 \
  pytest "models/autoports/zai_org_glm_4_7_flash/tests/test_long_context.py::test_full_context_202k" -q -s -m long
GLM47_DECODER=fused GLM47_REAL_WEIGHTS=1 GLM47_PCC_BAR=0.99 pytest ".../test_long_context.py::test_full_context_202k[dense]" -q -s -m long
GLM47_DECODER=fused GLM47_REAL_WEIGHTS=1 GLM47_EXPERT_DTYPE=bf4 GLM47_PCC_BAR=0.99 pytest ".../test_full_context_202k[moe]" -q -s -m long
# perf (tracy for the ops CSV, then standalone for clean walls)
python -m tracy -r -p -v -m pytest models/autoports/zai_org_glm_4_7_flash/tests/test_optimized_perf.py -q -s
pytest models/autoports/zai_org_glm_4_7_flash/tests/test_optimized_perf.py -q -s
tt-perf-report --arch p150 <ops_csv> --start-signpost PERF_DECODE_MOE_OPT --end-signpost PERF_DECODE_MOE_OPT_END --csv ...  # x8 windows
# watcher (separate run, no profiler)
TT_METAL_WATCHER=2 TT_METAL_LOGS_PATH=.../logs/watcher pytest .../test_optimized_decoder.py -q -s \
  -m "not real_weights" -k "decode_pcc or (prefill_pcc and moe-512) or cache_content or traced or bf8_cache"
# isolated-op sweeps (rerunnable; result tables in this log)
python models/autoports/zai_org_glm_4_7_flash/probe/optimize_perf_probe.py --roles absorbed,flat,norm,router,sparse,qpath,wopath,sparse_prefill
# dev loop driver
python models/autoports/zai_org_glm_4_7_flash/tests/dev_optimize.py --kind moe --decoder optimized --cache-dtype bf8 [--real] [--prefill]
```

Note: the isolated-op probe result JSONs from the sweep session were written
to /tmp and not preserved across the session boundary; every number is
recorded in this log's tables and the probe script is committed and
rerunnable.

## tt-perf-report advice pass on the final windows

- Decode rows: all dominant matmuls "in0_block_w looks good"; the "use
  HiFi2/HiFi4 for accuracy" note on the LoFi rows is the measured fidelity
  trade (HiFi2 arm +29 us/step decode; real-weight PCC passes at LoFi;
  rejected with evidence). "Place in0 in L1" on w_uv: its input is the flash
  output (DRAM per the SDPA memory config); the L1 variant is covered by the
  wo-batched/fold candidates (rejected table). dense decode "try DRAM-sharded
  for 2048x10240": tried, CB-clash at bw8 / slower legal configs / nw2 96 us
  vs 71.2 wide-1D (rejected with evidence).
- Prefill rows: "in0_block_w=1 is small" on the default-config flat matmuls
  -> ACTED ON: explicit 2D mcast configs (11x10, bw8, out-blocks capped 4x8
  for in-layer L1) for wqkv_a, wo, shared/dense gate/up/down. Isolated:
  wqkv 82->47, wo 235->134, dense down 452->247, shared gate 91->55 us per
  1024-chunk; in-layer: dense prefill 11.1 -> 10.3 ms, moe 125.8 -> ~125-127
  (flat rows are ~4% of the sparse-expert-dominated moe window). Small-M
  chunks keep the default config (2D grid would idle rows; preserves
  non-aligned prefill). First integration attempt hit a CB-vs-L1-tensor
  clash (program.cpp:1932) with per_core-sized out blocks; fixed by capping
  out_block_h/w - recorded as the L1-capacity adaptation, not a rejection.
  The per-head batched prefill matmuls (0.4-0.9 ms rows, 0.7% of the moe
  window) keep defaults: a MatmulMultiCoreReuse candidate needs
  per_core_M=32 blocks whose CBs exceed L1 and the rows are immaterial.
- Sparse rows: "utilization omitted (nnz=nullopt)" - reporting-only note
  (runtime-inferred nnz is deliberate: static nnz deadlocks on Blackhole
  zero-flush, tt-metal#45943/#45052, inherited decision).

All suites, the comparability ladder (5 passed), the deployment-arm 202k run
(bit-identical rows), and the watcher run were re-executed on the final code
after this change.

## Stage-review remediation (P1/P2 + concerns)

The first independent $stage-review returned more-work-needed with two
required items; both were remediated with hardware evidence:

P1 - mandatory dense/shared MLP BFP4/LoFi trial (previously dismissed with
text that conflated the shared-expert rows with the dense rows):
- Isolated bf4/LoFi rows (probe_results/probe_p1p2_bf4_sweep.log): dense
  gate 1D 65.3(bf8) -> 45.6(bf4); DRAM-sharded becomes legal at bf4 and wins
  (bw8 42.5, bw4 43.8; at bf8 it CB-clashed); dense down bw20 41.6 (bw10
  45.8, bw8 48.3, bw5 56.1); shared gate 1D 9.2; shared down bw6 10.5
  (bw3 14.7). All op-level PCC ~0.9936 (gaussian bf4 round-trip level).
- In-layer real weights: dense mlp-bf4 arm prefill509 0.99926 / decode
  0.9989+ / 0.4431 ms/tok; moe shared-bf4 arm prefill 0.99545 / decode
  0.9970+ / 0.4916 ms/tok.
- 202k real-weight decider: shared-bf4 KEPT (moe rows-at-bar 24/32 vs fused
  control 23/32, end agg 0.98822 vs 0.98752, decode@202751 0.99714 vs
  0.99778 - the one visible delta, above the 0.99 bar; win 8.8 us x46
  layers). dense-bf4 REJECTED (decode@202751 0.99865 vs 0.99993 control,
  end window 28/32 vs 30/32, for ~6.3 us on 1 of 47 layers; artifact
  long_context_dense_real_expertbf4_cachebf8_mlpbf4.json +
  logs/pytest_long_real_densemlpbf4_rejected.log). Final policy: shared
  expert bf4, dense MLP bf8 (split knobs mlp_*_dtype vs dense_mlp_dtype).

P2 - precision-locked bf4/LoFi geometry re-sweep for the rows the bf4
policy touches (probe_results/probe_p1p2_bf4_sweep.log):

| row | bf8-picked config | bf4-locked sweep | verdict |
|---|---|---|---|
| wqkv 8c | bw8 10.3 | bw4 15.0, 16c-bw4 15.1 | bw8 holds |
| wq_b 8c | bw3 12.5 | bw1 21.1; 4c bw6 12.0 isolated | 8c bw3 kept: the coupled 4c candidate is 11.97+5.68(4c norm)=17.65... corrected: 12.0+5.68=17.68 vs 12.5+5.40=17.88 us, i.e. ~0.2 us in the 4c arm's favor - inside the observed +-0.3 us probe run-to-run noise and 0.05% of the step; kept 8c (also keeps the q-norm grid uniform with the other 8c norms) |
| wo 8c | bw10 24.6 | bw5 29.7, **bw20 22.9**, 16c 24.7, 32c 30.0 | bw20 ADOPTED |
| w_uk 5x4 | bw6 osw4 | bw2/3/6 x osw2/4/8: 11.5-12.2 | unchanged (noise) |
| w_uv 5x4 | bw16 osw4 13.5 | **bw8 osw4 12.8** | bw8 ADOPTED |
| dense down | bw10 | **bw20 41.6** | bw20 in the (rejected) bf4 arm |

Other review items closed: long-context JSONs now carry an `arm` provenance
block and arm-suffixed filenames (no rerun can overwrite another arm);
decomposition arms rerun on the final code; deployment 202k run preserved
twice with bit-identical PCC rows (logs/pytest_long_real_deployment{,_rerun}
.log); committed osw>1 repro (probe --roles sparse_osw_bug: osw1 0.9939 OK /
osw2-3 0.822-0.870 CORRUPT); dense traced-decode PCC test and batch8+bf8-
cache cross test added to the suite; per-head prefill matmul justification
corrected (moe-window share ~1.6% of device time, dense-window ~19%,
model-level ~0.5%, exact L1/config-class blocker recorded); doc number
inconsistencies unified. Post-P1 addendum: dense-down bf8 geometry gap
closed (bf8 bw10 47.7 vs bw20 48.8 us -> bw10 stays the bf8 winner; bw20
only wins at bf4 - the cross is real in both directions); the committed
osw-bug repro has a preserved run log
(probe_results/probe_sparse_osw_bug.log); the perf harness, reports, and
the comparability ladder were re-executed after the final dense_mlp_dtype
flip so every "opt" artifact carries the final policy (dense decode rows
now show LoFi BF16 x BFP8; final walls moe 0.491 / dense 0.447 ms/tok,
moe device 333.7 us/step).

## Final numbers after remediation (same session, 8 windows)

moe decode 1.034 -> 0.491 ms/tok (device 787.1 -> 333.7 us/step); dense
0.969 -> 0.447; moe prefill 210.0 -> 125.5 ms; dense 15.2 -> 10.3 ms.
Roofline 78.4 us (40.2 MB/step) / device 333.7 / e2e 491 us. (The 0.443
dense number in the P1 section is the REJECTED bf4-dense arm; 0.447 is the
final-policy default run.)

## $optimize checklist status

- Decoder path fully traced, no host fallbacks: tripwire test + static audit pass.
- Decode activations width-sharded L1 across norm/attention/residual/MLP
  boundaries (residual on the 8-core DRAM-sharded-matmul raster; MoE glue
  interleaved-L1 — sparse_matmul does not support sharded output, gpt_oss idiom).
- Prefill activations DRAM interleaved; large prefill matmuls: sparse expert
  configs tuned (bw32/bw24); flat prefill matmuls HiFi2+fp32acc bf8 (default
  2D configs; prefill is flash+expert dominated, see prefill perf reports).
- Operation-topology audit: this log (audit table + candidate tables + actions).
- Best-candidate comparison: fusedbase re-measured same-session; every family
  (packed/split, DS/1D/reuse/batched-DS, nw1/nw2, fidelity, dtypes) has
  measured rows; final default reproduces the best candidates.
- Final default reproduced: wall JSONs regenerated on the final code
  (0.491/0.447 ms/tok) after the last policy change.
- Dtype/fidelity proven in measured rows: perf-report rows show
  `LoFi BF16 x BFP4/BFP8` on the dominant matmuls, `HiFi4 BF16 x FP32` router.
- SDPA/composite ops: flash MLA prefill/decode retained; SDPA config re-swept
  under bf8 cache (k128/mc8 kept); nlp_concat_heads_decode retained.
- Packed/fused same-input projections: wqkv_a packed (inherited, now
  DRAM-sharded); packed gate_up kept for routed experts; shared/dense packed
  gate_up measured and rejected (26.97 vs 26.75, 139.1 vs 137.6 us).
- memory_config/program_config/compute_kernel_config explicit on all material ops.
- Dominant-matmul geometry sweeps: per-role tables in this log (dram bw
  2..40, 1D grids 22..110 cores, sparse pcn/bw/osw cross with dtype per
  OPT-014); `in0_block_w<=2` nowhere in the final material rows (wq_b bw3 is
  K=768's 8-core maximum; 4-core bw6 measured slower 14.0 vs 13.1 us).
- Shard specs/core grids divide tiles cleanly (8-core raster = 8 banks;
  rect-grid helper enforces exact-fill).
- DRAM-sharded decode matmuls: wqkv_a, wq_b, wo, shared_down, dense_down.
- Collectives: n/a (single chip).
- MoE active-expert path: indexed sparse_matmul retained (batch-1), tuned
  geometry, L1 outputs, no dense all-expert runtime path, routing weights via
  embedding pick, expert reduction via sum (FastReduceNC row 2.4 us).
- LM head: n/a (decoder-layer stage).
- Reduced precision/fidelity: real-weight bf4 attention + bf4 experts + LoFi
  adopted; per-group tables above; broad frontier left to $datatype-sweep.
- Performance accounting reconciled: perf_summary.json.
- Batch capability: batch-1 primary; batch 8 + 32 union-path tests pass.
- Stress: 96-replay traced stress + bitwise determinism; watcher-clean run.

## Stage review outcome

- Round 1 (independent $stage-review subagent): more-work-needed (P1 dense/
  shared MLP bf4 trial missing + mischaracterized rejection text; P2
  precision-locked geometry re-sweep). Remediated with hardware evidence
  (sections above).
- Round 2: more-work-needed (dense perf artifacts captured under the
  rejected bf4-dense arm; stale contradictory doc text; git-exclude commit
  hazard; several doc nits). Remediated: full perf harness + reports +
  comparability ladder re-executed after the final policy flip (dense rows
  verified `LoFi BF16 x BFP8` in the shipped tracy tables); docs corrected.
- Round 3: **clean-pass** (no required work; two recommended one-line
  wording fixes applied before commit: the fused-control per-row comparison
  wording and stale checklist numbers). The reviewer re-derived the headline
  numbers, dtype rows, pytest summaries, and bytecode-diffed the cached pyc
  to prove the post-perf-run source edit was docstring-only.

## Log

- [x] Stage start: skills read, fused evidence reviewed, device smoke OK.
- [x] Topology audit written (above).
- [x] Reference idioms gathered (deepseek mla1d batched-DRAM-sharded contract,
      dram-sharded matmul validation, qwen36-blackhole wide-1D decode idiom,
      gpt_oss sparse configs + LoFi + L1 intermediates, sharded rmsnorm
      contract). Sparse matmul grid must be an exactly-filled rectangle;
      sharded output unsupported for sparse_matmul; gather/topk at [32,64]
      are single-core ops.
- [x] On-device candidate probe for dominant matmul roles (tables above).
- [x] optimized_decoder.py implementation (final policy above).
- [x] Correctness suite vs HF reference (synth 23 passed + real 3 passed,
      incl. deployment arm; batch 8/32; dense+moe traced + stress +
      determinism + bf8-cache arms).
- [x] Before/after perf (wall JSON x8 + tracy + tt-perf-report x8 windows).
- [x] Long-context ladder (comparability arm 5 passed) + real-weight
      deployment arms + fused controls; watcher-clean run; 96-replay stress.
- [x] README, perf_summary.json, context_contract update.
- [x] Stage review (3 rounds; final clean-pass), checkpoint commit.

## Checkpoint commit

- LOCAL CHECKPOINT COMMIT (never pushed), repo /home/stisi/tt-metal, branch
  ttmodelmanager/glm47-flash-probe, stage-owned paths only (101 files:
  tt/optimized_decoder.py, tests/test_optimized_{decoder,perf}.py,
  tests/dev_optimize.py, tests/test_long_context.py, probe/
  optimize_perf_probe.py, doc/optimized_decoder/**, doc/context_contract.json).
  Initial SHA 665f3ace2deb09e031640aa161d671442a355919; this work-log commit
  record is folded in by a single amend (same pattern as the previous
  stages) - the amended SHA is the one `git log` reports on the branch.
- Disk-only (repo 500 KB commit limit / previous-stage convention): the raw
  watcher.log (committed as .gz), the two >500 KB per-op decode perf-report
  .txt tables and three large inspector YAMLs (committed as .gz; the
  committed per-window CSVs carry every row), the raw full-session tracy
  ops CSVs under generated/profiler/reports/, and all __pycache__.
  NOTE: .git/info/exclude ignores models/autoports/ on this box; stage files
  are added with `git add -f` (both previous stages' files are tracked the
  same way).
- Pre-commit hooks (first commit attempt) reformatted tt/optimized_decoder.py
  (formatting-only); post-commit 6-test device smoke re-passed on the
  committed code (construction audit, decode PCC both kinds, cache content,
  moe+dense traced decode + determinism): 6 passed in 10s.
