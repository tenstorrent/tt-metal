# M3 SP=2 per-op study: ranked op roadmap for 16 x [2,4] prefill

Spec: `M3_SP2_OP_PROFILE_RUN.md`. All numbers come from the files in this directory (index at the end). Unless a
row says otherwise, figures are per sparse layer, averaged over layers 3-6, for a single prose request at
h = 139,264 on one (2,4) stage (SP=2, TP=4, EP=8).

## Summary

* **Lever #1 is `sparse_sdpa_msa`.** It takes 21-24% of the sparse layer's worst-chip time and runs at 3.6-4.0% eff. It re-fetches K/V for every query token (375-402 GB/s, DRAM-bound), and its time is flat in kv_len (2.88 ms at both 141k and 549k, 2048 rows). Raising it to target gives +17.3% / +16.7% goodput (near-term, p90 ≤ 10 s, W 4096 / 8192). Action: fork it for M3.
* **combine and moe_reduce mostly wait on the expert-hot chip.** 96% / 80% of their W=4096 worst-chip headroom is wait. Their kernel time is the fastest chip's time: moe_reduce 0.83 ms, which equals the bench. moe_reduce at target gives +7.8% / +5.1%; removing the wait alone gives +5.3% / +2.6%.
* **experts is already at 75-78% eff (mean chip) with ND + hybrid**, against Pavlo's 45%. The ND kernel fits max(weights, tokens), not the sum (R² 0.998 vs 0.971), so weight-read overlap is already in effect. The worst-chip gap is load imbalance (max/mean 1.5-2.1). The sim gains 0.0% from raising experts.
* **The h=0 (no-cache) MSA path gathers bf16 K/V.** sparse takes 5.25 ms there against 2.99 ms at 139k (W=4096). Feeding it bf8 saves about 2.2 ms per sparse layer (14%) on every request's first chunk.
* **The dense layers do not scan the whole slot capacity.** A 1M slot against a 64k slot changes 3 dense layers by -0.2 / +0.2 ms; a whole-capacity scan predicts +281 ms. Bounded dense gather is not a P0 item.
* **The blocking send is about 0.8 ms (0.7 / 0.8 / 0.9 ms at W 2048 / 4096 / 8192), not 7.5-28.5 ms.** The cold gap is stage compute. The async-handoff A/B (own links) hung on the first chunk twice, so it has no numbers.
* **misc:** per-segment ops add 0.31 / 0.61 ms per sparse layer at B=2 / 4. The rest of packed misc growth is an SP arrival wait behind the h=0 segment.
* **[4,2]:** KV PCC passes (≥ 0.9985 against [2,4]). Near-term goodput is -0.5% / -0.1% at 10 s and -10% / -26% at 3 s (native-gather estimate). The full stack reaches +3.9% / +8.7%, but it needs the uncommitted msa + async features, and the runner is not ready. **Verdict: keep [2,4].**
* **Fixing the top 5 ops together** gives +38.0% / +37.1% (near-term, 10 s). This is an upper bound, because it sets sparse and moe_reduce to target.

## Setup (short; details in Methods)

* **Branch.** `vmelnykov/m3_moe_fabric2d`, merged with main `c3177143` (merge `f3e7d4b860e`).
* **Packed-path gate after the merge: PASS.** On layers 0-4 the worst PCC is 0.99970, against the one-segment reference (`gate/gate_merge_L04_compare.txt`).
* **Fixed env.** `M3_MOE_W_NDSHARD=1`, `M3_MOE_HYBRID_THRESHOLD=128`, v1 dispatch/combine, bf4 experts, bf8 index_k cache.
  * The single-stage harness and the benches use 1d fabric.
  * The 4-rank runner needs 2d. With 1d it fails with `TT_FATAL !is_1d_fabric || !is_multi_mesh` (`control_plane.cpp:859`).
* **Tokens.** Real M3 ids: `longbook_56320` prose, plus an M3-tokenized code corpus built from the repo's `.py` files (`tools/make_code_corpus.py`; the token file is not checked in, the batches rebuild it).
* **Layers and depth.** Layers 0-6 run contiguously, so routing is real. At depth there is a real prefix fill (`PROFILE_PREFIX_QUIET=1`, captures of about 3 GB).
  * Single-request depth points are h = 139,264, which is 141,312 rounded down to a multiple of W.
  * Packed segments sit at h = 141,312.

## 1. Measured efficiencies, time shares and headroom ([2,4] sparse layer)

The columns are defined in `pavlo_table_ours.md`:

* **eff (mean)** = roof / (chip-mean ms − floor), which is Pavlo's form.
* **eff (worst)** = roof / worst-chip ms.
* **headroom** = target / min(eff mean, target).
* **share** = worst-chip op ms / worst-chip layer ms. The worst chips differ from op to op, so a column sums to more than 100%.
* Cells are W=4096 / W=8192. Pavlo's values are one 5120-token chunk at 51,200 cached.

| op | worst ms | roof ms | eff (mean) | Pavlo eff | eff (worst) | headroom | Pavlo hr | share | Pavlo share |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| sparse | 2.994 / 6.369 | 0.113 / 0.226 | 3.8 / 3.6% | 3.9% | 3.8 / 3.5% | x18.2 / x19.5 | x17.9 | 20.8 / 23.6% | 19.5% |
| experts | 2.759 / 4.237 | 1.453 / 1.911 | 78.3 / 74.8% | 44.7% | 52.7 / 45.1% | x1.0 / x1.0 | x1.6 | 19.1 / 15.6% | 18.9% |
| moe_reduce | 2.652 / 5.123 | 0.287 / 0.574 | 14.6 / 14.8% | 11.3% | 10.8 / 11.2% | x5.5 / x5.4 | x7.1 | 18.2 / 18.7% | 17.3% |
| combine | 2.022 / 3.808 | 0.252 / 0.503 | 31.9 / 33.7% | 29.5% | 12.4 / 13.2% | x2.5 / x2.4 | x2.7 | 13.9 / 13.9% | 6.0% |
| dispatch | 1.138 / 2.276 | 0.126 / 0.252 | 15.8 / 15.3% | 14.7% | 11.1 / 11.1% | x5.1 / x5.2 | x5.4 | 7.8 / 8.4% | 6.0% |
| norm_ag (x2) | 1.120 / 2.169 | 0.377 / 0.755 | 41.1 / 40.3% | 41.7% | 33.7 / 34.8% | x1.9 / x2.0 | x1.9 | 7.8 / 8.1% | 6.5% |
| shared | 0.972 / 1.829 | 0.379 / 0.759 | 41.8 / 43.5% | 42.0% | 39.1 / 41.5% | x1.7 / x1.6 | x1.7 | 6.8 / 6.8% | 6.3% |
| misc | 0.916 / 1.797 | 0.098 / 0.197 | 11.1 / 11.4% | 11.2% | 10.7 / 10.9% | x7.2 / x7.0 | x7.2 | 6.4 / 6.7% | 6.0% |
| indexer | 0.570 / 1.232 | 0.022 / 0.044 | 3.9 / 3.7% | 7.8% | 3.8 / 3.6% | x17.9 / x18.7 | x9.0 | 4.0 / 4.6% | 1.5% |
| attn_rs | 0.531 / 0.996 | 0.189 / 0.377 | 42.4 / 43.2% | 41.6% | 35.5 / 37.9% | x1.9 / x1.9 | x1.9 | 3.7 / 3.7% | 3.3% |
| ag_kv | 0.490 / 0.831 | 0.195 / 0.201 | 46.8 / 34.1% | 39.3% | 39.8 / 24.1% | x1.7 / x2.3 | x2.0 | 3.4 / 3.1% | 1.3% |
| idx_branch | 0.402 / 0.753 | 0.013 / 0.026 | 3.4 / 3.6% | 3.8% | 3.3 / 3.5% | x20.6 / x19.4 | x18.5 | 2.8 / 2.8% | 2.4% |
| o_proj | 0.312 / 0.590 | 0.170 / 0.339 | 56.5 / 58.8% | 58.5% | 54.4 / 57.5% | x1.2 / x1.2 | x1.2 | 2.2 / 2.2% | 2.0% |
| qkv | 0.251 / 0.448 | 0.191 / 0.381 | 79.6 / 87.3% | 87.6% | 76.0 / 85.1% | x1.0 / x1.0 | x1.0 | 1.8 / 1.7% | 1.5% |
| ag_idx | 0.224 / 0.232 | 0.097 / 0.100 | 54.2 / 53.9% | 100% (clamped) | 43.5 / 43.2% | x1.5 / x1.5 | x1.0 | 1.6 / 0.9% | 0.5% |
| router | 0.188 / 0.353 | 0.011 / 0.021 | 6.0 / 6.2% | 6.7% | 5.6 / 6.0% | x11.7 / x11.2 | x10.4 | 1.3 / 1.3% | 1.1% |
| **layer** | 14.40 / 27.00 (chip mean 14.21 / 26.54) | 3.97 / 6.67 | | Pavlo 18.58 ms | | | | | |

**Dense GQA layer (layer 1), W=4096 / 8192.** The layer's worst chip is 26.71 / 55.12 ms.

| op | worst ms | eff (mean) | Pavlo eff | headroom | share |
|---|---:|---:|---:|---:|---:|
| ring (ring_c) | 21.74 / 44.81 | 35.9 / 35.4% | 35.8% | x1.9 / x2.0 | 81.4 / 81.3% |
| dense_mlp | 1.68 / 3.24 | 59.8 / 61.7% | 61.1% | x1.2 / x1.1 | 6.3 / 5.9% |
| norm_ag (x2) | 1.15 / 2.25 | 41.3 / 39.6% | 43.3% | x1.9 / x2.0 | 4.3 / 4.1% |
| attn_rs | 1.02 / 2.75 | 31.7 / 26.4% | 41.7% | x2.5 / x3.0 | 3.8 / 5.0% |
| misc | 0.75 / 1.49 | 13.3 / 13.6% | 13.1% | x6.0 / x5.9 | 2.8 / 2.7% |
| o_proj / qkv | 0.31 / 0.58, 0.25 / 0.45 | 57-60%, 80-87% | 58.8%, 87.7% | x1.2, x1.0 | ≤ 1.2% |

The dense attn_rs worst chip is a wait: it absorbs the ring skew, against a floor of 0.44-0.46 ms. The mean-chip
`CAL.effs['2x4']` sets are in `tools/cal_effs_ours_2x4_w{4096,8192}.json`. For those, prose and code are pooled,
so experts is 76.9% / 74.5% there, against 78.3% / 74.8% prose-only above.

## 2. Ranked ops: headroom ms x share (single prose, h=139,264)

* headroom ms = max(0, worst ms − roof / target).
* key = headroom ms x share.
* min-chip hr applies the same formula to the fastest chip, so it is what is left once the cross-chip waits are removed.
* packed hr is the packed forward at the same W.

| op | 4k rank | 4k hr ms | 4k share | **4k key** | 4k min-chip hr | 4k packed hr | 8k rank | 8k hr ms | 8k share | **8k key** | 8k min-chip hr |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| sparse | 1 | 2.833 | 20.8% | **0.590** | 2.749 | 3.518 | 1 | 6.046 | 23.6% | **1.430** | 5.872 |
| moe_reduce | 2 | 2.293 | 18.2% | **0.417** | 0.469 | 1.449 | 2 | 4.405 | 18.7% | **0.825** | 0.925 |
| combine | 3 | 1.708 | 13.9% | **0.238** | 0.062 | 0.941 | 3 | 3.178 | 13.9% | **0.442** | 0.020 |
| experts | 4 | 0.683 | 19.1% | **0.131** | 0.000 | 0.311 | 4 | 1.508 | 15.6% | **0.236** | 0.000 |
| dispatch | 5 | 0.980 | 7.8% | **0.077** | 0.562 | 0.807 | 5 | 1.961 | 8.4% | **0.164** | 1.112 |
| misc | 6 | 0.793 | 6.4% | 0.050 | 0.756 | 2.330 | 6 | 1.551 | 6.7% | 0.103 | 1.445 |
| norm_ag | 7 | 0.648 | 7.8% | 0.050 | 0.445 | 0.605 | 7 | 1.225 | 8.1% | 0.099 | 0.873 |
| shared | 8 | 0.429 | 6.8% | 0.029 | 0.384 | 0.435 | 9 | 0.744 | 6.8% | 0.051 | 0.669 |
| indexer | 9 | 0.539 | 4.0% | 0.021 | 0.535 | 0.345 | 8 | 1.169 | 4.6% | 0.053 | 1.088 |
| attn_rs | 10 | 0.295 | 3.7% | 0.011 | 0.212 | 0.253 | 11 | 0.524 | 3.7% | 0.019 | 0.394 |
| idx_branch | 11 | 0.383 | 2.8% | 0.011 | 0.378 | 0.436 | 10 | 0.715 | 2.8% | 0.020 | 0.701 |
| ag_kv | 12 | 0.246 | 3.4% | 0.008 | 0.190 | 0.323 | 12 | 0.580 | 3.1% | 0.018 | 0.195 |
| router | 13 | 0.173 | 1.3% | 0.002 | 0.170 | 0.172 | 13 | 0.323 | 1.3% | 0.004 | 0.316 |
| ag_idx | 14 | 0.102 | 1.6% | 0.002 | 0.094 | 0.123 | 15 | 0.107 | 0.9% | 0.001 | 0.097 |
| o_proj | 15 | 0.070 | 2.2% | 0.002 | 0.066 | 0.071 | 14 | 0.106 | 2.2% | 0.002 | 0.099 |
| qkv | 16 | 0.000 | 1.8% | 0.000 | 0.000 | 0.000 | 16 | 0.000 | 1.7% | 0.000 | 0.000 |

How to read the ranks:

* **Ranks 2-4 (moe_reduce, combine, experts) are mostly imbalance.** Their min-chip headroom is small next to their worst-chip headroom (section 7).
* **sparse, dispatch, misc, norm_ag, shared and indexer** have real kernel headroom of 0.4-0.8 ms each at W=4096. For sparse it is 2.8 ms.

## 3. Goodput per fix (Pavlo's sim, 16 x [2,4], our calibration)

**How it was run.**

* The base is `cal_effs_ours_2x4_w{W}.json` with our `pipe.ringC`.
* Each row merges one fix file from `goodput/fix/` and reruns the config.
* Δ is against the base at the same features, W and SLO. Source: `goodput_results.md` section 3.

**How to read it.** The sim was run at two fix values: the target, and the min-chip eff (the wait removed). The
"realistic reach" column is our estimate from the op studies. It was not run through the sim.

| fix | realistic reach (estimate) | near 4k 10 s | near 4k 3 s | near 8k 10 s | near 8k 3 s | full 4k 10 s | full 8k 10 s |
|---|---|---:|---:|---:|---:|---:|---:|
| base goodput | | 29.7k | 24.0k | 30.4k | 20.8k | 40.6k | 42.0k |
| 1. sparse 3.6-4.0% → 70% | 4-8x K/V reuse gives 0.36-0.72 ms of K/V traffic at 2048 rows (`msa_summary.md`), about 15-30% eff, so this row is an upper bound | **+17.3%** | +24.5% | **+16.7%** | +36.9% | +14.8% | +14.9% |
| 2. moe_reduce 17% → 80% | the kernel fix takes post_combine_reduce from 0.36 to about 0.1 ms; the RS stays at about 40% of link | **+7.8%** | +11.4% | **+5.1%** | +12.8% | +5.1% | +7.0% |
| 2'. moe_reduce → min-chip 36% | the wait removed only; needs expert balance, not a kernel change | +5.3% | +8.2% | +2.6% | +6.1% | +2.7% | +5.1% |
| 3. combine 38-39% → min-chip 79 / 80% | the whole gain is the wait; the fastest chip is already at target | **+3.1%** | +4.1% | **+1.3%** | +2.2% | +0.9% | +1.1% |
| 4. experts → 70% | already above target, so the sim clips it | **+0.0%** | +0.0% | **+0.0%** | +0.0% | +0.0% | +0.0% |
| 4'. experts roofline imbalance 1.2 → 1.0 | perfect balance | -0.2% | +0.2% | +0.1% | +0.4% | +0.3% | +0.7% |
| 5. dispatch 16% → 80% | tune; no measured realistic value | **+3.7%** | +6.1% | **+2.1%** | +3.9% | +1.8% | +2.9% |
| 5'. dispatch → min-chip 18.6% | | +2.7% | -0.2% | -0.1% | +0.4% | +0.9% | -0.1% |
| all 5 (rows 1-5) | upper bound | **+38.0%** | +45.0% | **+37.1%** | +65.0% | +27.5% | +29.7% |

* **The fixes compound.** "All 5" is more than the sum of the singles, because once sparse is fixed the MoE chain becomes a larger share of the critical stage.
* **3 s cells are about ±5% resolution.** That SLO is a cliff regime (section 6).
* **The dispatch rows leave `kv_a2a` unchanged**, although kv_a2a is a copy of dispatch's eff.

## 4. One-line action per op

| op | action | class |
|---|---|---|
| sparse | Fork `sparse_sdpa_msa` for M3. Amortise K/V across consecutive query tokens: tile tokens per group over the union of their selected blocks, and pack 2 tokens x 16 heads per 32-row tile. Also feed the h=0 path from bf8 K/V (section 8). | **fork** |
| moe_reduce | Fork the `post_combine_reduce` reader so it skips non-local slots (3 of 4 are skipped in compute but still read), keeps more than one row in flight, and uses all 120 cores. The 1.2-3.6 ms RS wait goes only with expert balance. | **fork** (+ balance) |
| combine | No kernel work: the fastest chip is at 79-87%. The wait is removed by expert placement or replication, capacity-aware routing, or packing. | balance, not kernel |
| experts | No kernel work at the operating point: 75-78% mean, above the 70% target. The worst-chip gap is imbalance. | balance, not kernel |
| dispatch | Tune (16%). The extra traffic on the hot column is real, load-proportional traffic that both chips of the pair pay. | **tune** |
| misc | Batched RoPE (slice / rotary / concat), q/k norm, head split/concat and KV / index_k write across segments. Needs segment offsets passed to one whole-tensor op. | **needs batching metadata** |
| norm_ag | Tune the CCL config (41% against 80%). | **tune** |
| shared | Tune the matmul program config (42% against 70%). | **tune** |
| indexer | Tune the program config (96 cores, 25-29% of LoFi on its true FLOPs; about 2.4-2.8x headroom). Going further needs block-pooled index keys, which is an algorithm change. | **tune** |
| attn_rs | Tune. It is the same RS collective as in moe_reduce and shared, at about 40% of link. | **tune** |
| idx_branch | Fold the index q/k projections into `qkv_proj` (today the matmul is N = 4 tiles, at about 7% of HiFi2), then fuse norm + RoPE. This is a model-level change. | **tune** (model-level) |
| ag_kv | Tune (34-47%). In packed forwards it runs once per segment. | **tune** + batching metadata |
| MSA chain in packed forwards (idx_branch, ag_*, indexer, sparse) | Runs one chain per segment. | **needs batching metadata** |
| router | Tune (6%, but 1.3% share, so low priority). | tune (low) |
| ag_idx, o_proj, qkv | At or near target, or ≤ 1.6% share. | none |
| dense ring (ring_joint) | Tune (36% against 70%; 81% of the dense layer at 139k). No capacity-bound gather is needed (section 10). | **tune** |
| dense attn_rs | A wait on the ring skew; it goes with the ring. | none |
| dense_mlp | 60-62% against 70%. | tune (low) |

## 5. What changed vs Pavlo's calibration

| item | Pavlo | ours (W 4096 / 8192) | why |
|---|---|---|---|
| shape | 5120 tok @ 51,200 | 4096 / 8192 @ 139,264 | Token-proportional ops keep their eff (within ±2 pts); qkv is 79.6% at 4096 because the weight read is a larger part. |
| experts | 44.7% | 76.9 / 74.5% (mean chip, pooled) | ND-sharded weights + hybrid threshold 128. |
| combine | 29.5% | 38.0 / 39.3% | The mean still contains the waits of 7 chips. Worst chip is 2.4-2.5x the mean (Pavlo's own note: 2.4). |
| moe_reduce | 11.3% | 17.0 / 17.4% | Same: the mean contains the RS wait. The kernel on the min chip is 0.83 ms, equal to the bench. |
| indexer | 7.8% | 4.1 / 3.7% (2.3% at 549k) | The kernel computes the full rows x kv_len x 128 rectangle. The sim roofline models pooled scoring and grows 32x slower with depth. |
| ag_idx | 100% (clamped) | 54% | At 51k his roofline sat above measured − floor. At 139k it is measurable. Our roofline uses bf8 index_k (1.0625 B); he profiled bf16. |
| ag_kv | 39.3% | 45.4 / 33.7% | It moves the persistent gather buffer. Layer 3 adds an SP arrival wait behind the dense ring (1.6 ms on rank 0 at 8192). |
| dense attn_rs | 41.7% | 34.1 / 26.3% | Wait on the ring skew. |
| routing | not modelled | max/mean chip load 2.1 prose, 1.65 code, 1.5-1.6 packed | The sim assumes a fixed imbalance of 1.2 in the experts roofline. |
| h=0 path | not seen | sparse 5.25 ms against 2.99 ms at 139k | The no-cache MSA path uses bf16 K/V (section 8). |
| unchanged | | norm_ag, attn_rs, shared, o_proj, router, sparse, idx_branch, misc, dense ring, dense_mlp within 1-2 pts | Same mesh, same 1D fabric and Linear CCLs. |

## 6. [2,4] vs [4,2]

### Per-op efficiency side by side (mean chip, Pavlo form; `pavlo_table_ours_4x2.md`, `goodput_results.md` section 1)

(4,2) was profiled on prose only. The "native" column removes the harness per-head slice/concat copies from ag_kv.

| op | [4,2] W4096 | [4,2] W8192 | Pavlo [4,2] | [2,4] W4096 / W8192 | target |
|---|---:|---:|---:|---:|---:|
| norm_ag | 54.2% | 47.3% | 51.1% | 40.9 / 40.1% | 80% |
| qkv | 72.0% | 80.3% | 87.2% | 79.6 / 87.3% | 70% |
| idx_branch | 5.7% | 6.6% | 6.2% | 3.4 / 3.6% | 70% |
| o_proj | 56.5% | 66.2% | 66.7% | 56.5 / 58.8% | 70% |
| attn_rs | 45.4% | 43.0% | 44.6% | 42.7 / 43.0% | 80% |
| shared | 49.3% | 53.7% | 52.7% | 41.9 / 43.5% | 70% |
| router | 5.5% | 6.0% | 5.0% | 6.0 / 6.2% | 70% |
| dispatch | 27.3% | 27.1% | 29.1% | 16.4 / 16.1% | 80% |
| experts | 78.1% | 74.8% | 44.3% | 76.9 / 74.5% | 70% |
| combine | 33.2% | 34.1% | 22.7% | 38.0 / 39.3% | 80% |
| moe_reduce | 16.1% | 16.1% | 17.6% | 17.0 / 17.4% | 80% |
| ag_kv (copies / native) | 30.1 / 41.1% | 28.0 / 37.1% | 26.7% | 45.4 / 33.7% | 80% |
| ag_idx | 46.8% | 46.8% | 100% | 54.1 / 53.9% | 80% |
| indexer | 2.1% | 1.9% | 4.2% | 4.1 / 3.7% | 70% |
| sparse | 4.1% | 3.9% | 3.9% | 4.0 / 3.6% | 70% |
| misc | 11.1% | 14.3% | 12.0% | 11.2 / 11.4% | 80% |
| dense ring_c | 31.2% | 35.0% | 35.5% | 37.4 / 35.0% | 70% |

**Worst-chip time ratio [4,2] / [2,4]** at h=139k, W 4096 / 8192 (`compare_4x2_vs_2x4.txt`):

* **The TP collectives get cheaper on 2 chips:** norm_ag 0.28 / 0.29, attn_rs 0.43 / 0.38, shared 0.58 / 0.55, router 0.57 / 0.54 and idx_branch 0.61 / 0.55.
* **qkv costs more** (1.10 / 1.09): each chip reads twice the weights.
* **ag_kv costs much more:** 4.29 / 2.98 as measured, about 3.2 / 2.3 with the native-gather estimate. The 3x in the roofline is structural: each chip gathers 3/4 x kv_len for 2 heads, against 1/2 x kv_len for 1 head.
* **The MoE chain shifts cost.** combine is 1.26 / 1.28, because dispatch and combine run Linear over 4 hops. moe_reduce is 0.65 / 0.67, because the RS runs over 2 chips.
* **experts, sparse and indexer are about 1.0.**
* **The dense ring** is 1.20 / 1.00 (1024 / 2048 rows per chip) and 1.34 / 1.32 in packed forwards (512 rows per chip per 2048-token segment). So the small-row penalty comes back at 1024 and at 512 rows per chip.
* **The torus MoE ops don't apply:** a carved (4,2) has no wrap link, so only v1 ran.

**Chip efficiency.** Chip-µs per token-layer (layer worst ms x 8 / W), [4,2] / [2,4]:

| layer | W | h=0 | h=139k | h=549k | packed |
|---|---:|---:|---:|---:|---:|
| sparse | 4096 | 27.1 / 30.4 | 27.8 / 28.1 | 43.8 / 34.5 | 28.4 / 28.3 |
| sparse | 8192 | 25.6 / 29.6 | 23.8 / 26.4 | 37.3 / 30.7 | 36.3 / 29.1 |
| dense | 4096 | 7.9 / 9.9 | 58.0 / 52.2 | 206.0 / 176.9 | 41.6 / 35.2 |
| dense | 8192 | 7.9 / 10.7 | 49.8 / 53.8 | 178.8 / 183.5 | 91.4 / 73.9 |

[4,2] wins at shallow depth on sparse layers, from the cheaper TP collectives. It loses at 549k and in packed
forwards, because of ag_kv (8.2 / 12.8 ms against 1.85 / 1.94 ms) and the small-row dense ring.

### Correctness on (4,2) and the KV-head sharding change (`kv_4x2_status.md`)

* **PCC.** With harness patches only (`tools/profile_4x2.py`), layers 0-6, 5120 at 51,200:
  * min PCC against the golden is 0.99524 on (4,2), against 0.99531 on (2,4);
  * (4,2) against (2,4) is ≥ 0.9985;
  * layer 0 is bit-identical.
  * **Status: correct.**
* **What breaks at TP=2 today:**
  * `kv_cache.py` allocates 1 K/V head per chip;
  * the MSA cache read uses the selected-batch `high_bw_all_gather`, which TT_FATALs on a non-singleton head dim;
  * `kv_chunk_table.py:96` asserts `num_kv_heads == cols`.
* **Work to make it production:**
  * (a) Upstream the patches plus (4,2) tests with num_groups=2: 1-1.5 d.
  * (b) A native multi-head gather that removes the head copies: 2-3 d. The copies cost 0.53 ms (3.6%) at 4k / 139k and 6.16 ms (16.6%) at 8k packed.
  * (c) Runner and KV migration: 1-2 d.
  * **Total: about 4-6.5 d.** Only (a) is needed for correctness.
* **Runner blockers** (`recipe_4x2.md`): there is no SP=4/TP=2 topology yaml, no chained 4-mesh [4,2] mesh graph descriptor, no per-tray `TT_VISIBLE_DEVICES`, no full 60-layer [4,2] weight cache (about 45 min of conversion), and 2d fabric on 4x2 has not been validated. For these reasons the optional 4 x [4,2] pipeline run (P1-D) and the middle-rows torus run (P2) were skipped.

### Goodput, 16 x [4,2] vs 16 x [2,4] (`goodput_results.md` section 2)

Cells are [2,4] / [4,2] in k tok/s, with Δ for [4,2]:

* `near` = pool bounded host async batch idxdedup, which is P0+P1 without the variable chunk;
* `full` = the g4_k0 study stack.

| calibration | features | W 4096, p90 ≤ 10 s | W 4096, ≤ 3 s | W 8192, ≤ 10 s | W 8192, ≤ 3 s |
|---|---|---|---|---|---|
| Pavlo's | near | 28.4 / 27.5 (-3.2%) | 20.1 / 16.4 (-18.1%) | 29.3 / 28.2 (-3.6%) | 15.4 / 8.1 (-47.3%) |
| Pavlo's | full | 38.5 / 40.1 (+4.2%) | 27.8 / 30.7 (+10.5%) | 39.8 / 42.5 (+6.9%) | 24.6 / 29.9 (+21.6%) |
| ours, [4,2] with copies | near | 29.7 / 28.9 (-2.4%) | 24.0 / 20.8 (-13.3%) | 30.4 / 29.4 (-3.3%) | 20.8 / 8.8 (-57.7%) |
| ours, [4,2] with copies | full | 40.6 / 41.9 (+3.2%) | 31.2 / 37.0 (+18.4%) | 42.0 / 45.4 (+8.1%) | 28.9 / 32.7 (+12.9%) |
| ours, [4,2] native gather | near | 29.7 / 29.5 (**-0.5%**) | 24.0 / 21.5 (-10.3%) | 30.4 / 30.4 (**-0.1%**) | 20.8 / 15.4 (-26.1%) |
| ours, [4,2] native gather | full | 40.6 / 42.2 (+3.9%) | 31.2 / 36.0 (+15.2%) | 42.0 / 45.6 (+8.7%) | 28.9 / 34.5 (+19.3%) |

* **Pavlo's "near-term" set is not defined in his material.** `near` is the closest match: -3.2% / -3.6% at 10 s against his "2-3% worse". It matches only at the 10 s SLO (`pavlo_reference.md`).
* **The 3 s near-term cells are a cliff regime.** The unloaded p90 is already 2.0-2.3 s, so read them at about ±5%.
* **The supplementary `p0p1` set (near + var), native:** -0.6% / +9.6% at 10 s (W 4096 / 8192).
* **Sensitivity checks:** a prose-only [2,4] calibration, or Pavlo's `pipe.ringC`, moves the near / 10 s cells by ≤ 0.4 pts.

### Decision rule verdict: **keep [2,4]**

| rule | status | evidence |
|---|---|---|
| (a) KV PCC on (4,2) passes, or the sharding change is ≤ 1-2 weeks | **pass** | PCC ≥ 0.9985 against (2,4); the change is scoped at 4-6.5 d |
| (b) ≥ +5% at the SLO with near-term features | **fail** | -0.5% / -0.1% at 10 s and -10% / -26% at 3 s (native estimate); -2.4% / -3.3% with the copies |
| (b') or ≥ +5% with the full stack, if msa + async are committed | not applicable | Full stack: +3.2-3.9% at 4k / 10 s, +8.1-8.7% at 8k / 10 s. But `msa` (the SP-local indexer) and `async` are sim-only, and the runner's async-handoff knob hangs (section 9). |
| (c) The runner and KV migration handle TP=2 without new blockers | **fail** | The runner pieces above are missing; `kv_chunk_table.py:96` asserts. |

**What would reopen [4,2]:** msa + async committed on the roadmap, plus the runner and migration work scoped. Under
those conditions [4,2] leads by about 9% at 8k / 10 s and by 13-19% at 3 s in the full stack.

## 7. MoE chain on real routing (P0-B; `moe_chain_summary.md`)

**Expert load, max/mean over the 8 chips** (`load_skew.csv`, mean of layers 3-6):

| case | max/mean | hottest expert's share |
|---|---:|---:|
| (a) single prose, h=0 / h=139k | 2.05 / 2.15 | 13.3 / 14.4% |
| (b) single code, h=0 / h=139k | 1.61 / 1.68 | 10.0 / 10.1% |
| (c) packed W=4096 (2 segments) / W=8192 (4 segments) | 1.62 / 1.53 | 8.7 / 8.8% |

* **Packing reduces the skew against prose (2.1 to 1.5-1.6), but not against code.**
* **Worst case: prose layer 4.** One expert takes 22-23% of all assignments, and chip (0,2) gets 2.9x the mean.

**Per-chip chain, worst / mean / min ms:**

| op | W=4096 prose | W=4096 packed | W=8192 prose | W=8192 packed |
|---|---|---|---|---|
| dispatch | 1.138 / 0.838 / 0.719 | 0.964 / 0.803 / 0.727 | 2.276 / 1.682 / 1.426 | 1.812 / 1.569 / 1.446 |
| experts | 2.759 / 1.865 / 1.444 | 2.387 / 1.964 / 1.658 | 4.238 / 2.563 / 1.764 | 3.342 / 2.612 / 2.087 |
| combine | 2.022 / 0.828 / 0.377 | 1.255 / 0.653 / 0.368 | 3.808 / 1.535 / 0.650 | 2.374 / 1.233 / 0.685 |
| moe_reduce | 2.652 / 2.011 / 0.828 | 1.808 / 1.362 / 0.837 | 5.123 / 3.910 / 1.643 | 3.222 / 2.398 / 1.652 |
| chain max, balanced estimate | 5.63, 3.79 | 4.83, 3.90 | 9.85, 6.28 | 7.95, 6.40 |
| **imbalance cost of the chain** | **1.84 ms (33%)** | 0.94 (19%) | **3.57 (36%)** | 1.55 (20%) |

**Where the worst-chip time goes.** Counts are over 24 (run, layer) points:

* **The worst experts chip** is the hot chip in 20 of 24 points.
* **The worst combine chip** is the hot chip's SP partner in 21 of 24. The combine writer's init handshake waits for the peer to leave experts.
* **The worst moe_reduce chip** is never in the hot column. The RS on axis 1 waits there for the late row peer.
* **The kernel is the fastest chip.** moe_reduce min is 0.83 ms (bench fused + RS: 0.805-0.851 ms at 2048 tokens per chip); at 8192 it is 1.64 ms, against 1.61-1.68 in the bench.

**experts: imbalance against kernel** (`load_skew_join.csv`, bench hybrid fit):

| case | worst ms | mean ms | imbalance (worst − mean) | predicted by the bench from counts | residual |
|---|---:|---:|---:|---:|---:|
| prose, W=4096 | 2.70 | 1.90 | 0.80 (30%) | 0.65 | 0.34 |
| code, W=4096 | 2.40 | 1.95 | 0.46 (19%) | 0.37 | 0.25 |
| packed, W=4096 | 2.39 | 1.96 | 0.42 (18%) | 0.36 | 0.23 |
| packed, W=8192 | 3.34 | 2.61 | 0.73 (22%) | 0.59 | 0.38 |

* **The worst chip's experts time is 70-82% kernel and 18-30% imbalance.**
* **The chain pays the imbalance about twice.** Its imbalance cost is about 2x experts' worst minus mean.

**experts kernel model** (`experts_fit.txt`, 16 experts per chip, 16-1024 tokens per expert x 4 / 8 / 16 active):

| path | better fit | R² sum / max (no +c) | parameters |
|---|---|---:|---|
| nd (`unified_routed_expert_moe`) | **max(a·W, b·T)** | 0.971 / **0.998** | max+c: 259 GB/s effective weight read, 0.361 µs per token, c = 31 µs |
| hybrid (threshold 128, deployed) | sum, by a small margin | 0.989 / 0.985 | additive+c: 427 GB/s, 0.274 µs per token, c = 68 µs |

* **The ND kernel already overlaps the weight read with compute.** Pavlo's "read plus compute" model is refuted for it.
* **The hybrid path fits the sum slightly better.** But at the balanced operating point (128 tokens per expert x 16 active, T = 2048) it takes the same time as ND: 1.955 against 1.939 ms.
* **Consequence.** An "overlap weight reads" fix has no target left at the operating point. The sim clips experts at 70% in any case (section 3, row 4).
* **Disagreement with two source files.** `moe_chain_summary.md` and `pavlo_table_ours.md` conclude from the hybrid fit that "overlap is the kernel fix". We read the same fit differently: the hybrid margin is small (R² 0.990 against 0.986 with +c), and ND already overlaps at equal time.

**moe_reduce microbench** (`moe_reduce_bench.csv`, top-k 4, emb 6144):

| tokens per chip | fused ms | bytes read / GB/s | useful GB/s | cores | RS ms | fused + RS ms |
|---:|---:|---:|---:|---:|---:|---:|
| 1024 | 0.192 | 62.9 MB / 327 | 131 | 32 | 0.224-0.262 | 0.415-0.454 |
| 2048 | 0.367 | 125.9 MB / 343 | 137 | 64 | 0.438-0.493 | 0.805-0.851 |
| 4096 | 0.750 | 251.7 MB / 336 | 134 | 120 | 0.868-0.942 | 1.607-1.676 |

* **Per-core loop:** `for chunk: for token(32): for slot(4)`. The reader does one 12 KB read per slot, serialised with a read barrier and one row in flight, and skips no slot. About 1 slot in 4 is local.
* **The core grid is min(tokens/32, grid).**
* **The reader moves 4x the useful bytes at about 340 GB/s.**
* **The RS runs at about 40% of its link roofline.**

## 8. MSA ops at M3 shapes (P0-C; `msa_summary.md`, `msa_bench.csv`)

**Bench** (worst chip ms, 16 q heads, 1 KV group and 1 index head per chip, top-16 blocks of 128, bf8 K/V):

| rows/chip, kv_len | sparse_sdpa_msa | indexer_score_msa | topk | idx_branch | full msa chain |
|---|---:|---:|---:|---:|---:|
| 1024, 141k | 1.500 | 0.241 | 0.050 | 0.223 | 1.854 |
| 2048, 4k | 2.935 | 0.094 | 0.041 | 0.402 | 3.204 |
| 2048, 141k | 2.879 | 0.472 | 0.095 | 0.402 | 3.563 |
| 2048, 549k | 2.878 | 1.605 | 0.264 | 0.402 | 4.842 |
| 4096, 141k | 6.116 | 0.943 | 0.184 | 0.751 | 7.465 |
| 4096, 549k | 6.123 | 3.244 | 0.513 | 0.750 | 10.067 |

| op | scales with | cores | achieved | bound / cause | in-model vs bench |
|---|---|---|---|---|---|
| sparse_sdpa_msa | rows only; flat in kv_len; not a latency floor | 120 / 120, even split of (token, group) items; not serial over heads | 11-12 TFLOP/s, **375-402 GB/s (73-79% of DRAM)** | Gather-bound: each query token re-reads 16 x 128 keys x (K+V), about 557 KB, with no sharing between tokens. The roofline assumes 32x reuse, which is the whole 3.9%. | +0.125 ms of layout ops |
| indexer_score_msa | rows x kv_len (3.4x from 141k to 549k) | 96 (32 at kv = 4k) | **154-179 TFLOP/s on the true rectangle** (25-29% of LoFi) | Computes the full [0, kv_len) rectangle, with no causal saving. The sim roofline models pooled scoring, 11-22x fewer FLOPs. | within 1-4% |
| topk_large_indices | rows x (kv/128), sub-linear | 120 | 47-69 GB/s | 14-17% of the indexer zone | in `indexer` |
| idx_branch | rows only | 120 | 14-17 TFLOP/s | Two N = 4-tile matmuls (about 0.157 ms each, about 7% of HiFi2) plus 16 small ops (about 0.085 ms). The true roofline is about 0.021 ms (256 columns, not 160). | equal |

**Anomaly B: the h=0 path.**

* **What happens.** With `cached_len == 0`, `msa_sp_attention_nocache` all-gathers fresh **bf16** K/V. At depth the op reads the bf8 cache instead.
* **Cost.**
  * sparse on rank 1 takes 5.25 ms (W=4096) and 11.04 ms (W=8192), against 2.99 and 6.37 ms at 139k. That is 1.74x the bench at the same shape.
  * SP rank 0 finishes about 2 ms early and waits in the MoE routing-setup all-gather: 2.05-2.08 ms, charged to misc.
* **Fix.** Typecast before the gather, or read the just-written cache. It saves about 2.2 ms per sparse layer (14%) at W=4096 and about 4.6 ms (15%) at W=8192 on every first chunk.

## 9. Pipeline overheads on 4 x (2,4) (P1-D; `pipeline_overheads.txt`)

**Setup.** The common runner, 60 layers split 15/15/15/15, 2d fabric, lease mode (share_fabric_links=1).
`PREFILL_SYNC_PER_CHUNK=1` gives the per-stage compute. "Hot" is a synthetic 139,264-token prefix, so the history
KV is never written. Medians are per stage, n = 24.

| W | blocking send (push + lease_out) | Pavlo's block fit | hop (receiver waiting) | host gap | bottleneck compute, cold / hot | runner steady tok/s, cold / hot | compute-bound tok/s, cold / hot |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 2048 | 0.69-0.78 (hot up to 1.22) ms | 7.5 ms | 4.9-7.8 ms | 0.33-0.43 ms | 135.5 / 166.0 ms | 14,451 / 11,821 | 15,110 / 12,335 |
| 4096 | 0.78-0.88 ms | 14.5 ms | 9.5-10.8 ms | 0.41-0.59 ms | 260.6 / 275.9 ms | 15,051 / 14,372 | 15,715 / 14,848 |
| 8192 | 0.81-0.98 ms | 28.5 ms | 18.7-20.9 ms | 0.43-0.56 ms | 491.2 / 521.2 ms | 15,974 / 14,905 | 16,678 / 15,717 |

* **The blocking send is not the cold gap.** It is about 0.3% of a stage.
  * The old SP2 cold gap (-12.4%: 15.7k measured against 17.9k predicted) is stage compute. In the pipeline the sparse stages take 247 / 261 / 247 ms (W=4096, cold), which alone gives 15.7k.
  * The un-synced lease pipeline runs 15,225 / 14,425 tok/s steady. That is -3.1% / -2.9% against the sync compute bound, and the difference is input wait and handoff on the bottleneck stage.
* **The hop is about 2.4 ms per 1k tokens.** It is latency on an idle receiver, not sender-busy time. It matches the magnitude of Pavlo's "send".
* **The async-handoff A/B (`PREFILL_D2D_SHARE_FABRIC_LINKS=0`, own links) hangs on the first chunk, twice** (`ab_own_w4096_sync0`, `pcc_own_w4096_sync0`). Stage 1 never receives chunk 0. There are no timing or PCC numbers for the async handoff.
* **The 1d fabric fails at startup on the 4-rank runner** (`d_w4096_sync1_fab1d`), so every session uses 2d.
* **KV read-back on the runner (lease mode):**
  * min PCC is 0.8846 (V) and 0.9646 (K);
  * both decay smoothly over layers 0-59, with no step at the stage boundaries, so the handoff is clean;
  * the 0.93 default threshold is too tight for 60 layers with bf4 experts.
* **The dense-heavy split `1,1,1,57` was not run** (60 layers loaded fine, so the fallback wasn't needed).

## 10. Dense scan (P1-E; `dense_scan_check.txt`)

| harness | point | 64k slot | 1M slot | Δ | whole-capacity scan predicts |
|---|---|---:|---:|---:|---:|
| budget_sweep, dense layers 0-2 | h=0, n=4096 | 16.74 ms | 16.97 ms | +0.23 ms (+1.4%) | +281 ms |
| budget_sweep, dense layers 0-2 | h=16384, n=4096 | 26.23 ms | 26.02 ms | -0.21 ms (-0.8%) | +281 ms |
| runner, 1 dense layer per stage (s0-s2) | cold | 8.35 / 5.77 / 5.99 | 8.36 / 6.96 / 6.94 | +0.0 to +1.2 ms | +94 ms per stage |
| runner, 1 dense layer per stage (s0-s2) | h=16k | 10.74 / 8.80 / 9.53 | 11.34 / 9.99 / 10.07 | +0.5 to +1.2 ms | +94 ms per stage |

* **Verdict: there is no whole-capacity scan.** Dense cost follows the history h, not the capacity.
* **The 1M slot was really allocated:** DRAM per bank went from 0.16 to 0.27 GiB.
* **Bounded dense gather is not P0.**

## 11. misc (P1-F; `misc_breakdown.txt` / `.csv`)

**Single request, h=139k, chip-mean ms per layer:**

| group | sparse W=4096 | sparse W=8192 | dense W=4096 | tag |
|---|---:|---:|---:|---|
| norm (input + post-attention LayerNorm) | 0.313 | 0.620 | 0.310 | whole tensor |
| rope (q and k: slice / rotary / concat, 8 ops) | 0.177 | 0.341 | 0.177 | per segment |
| residual adds (3 sparse, 2 dense) | 0.146 | 0.288 | 0.092 | whole tensor |
| heads (split / concat) | 0.099 | 0.218 | 0.095 | per segment |
| routing_setup (AllGather, MaskedBincount, OffsetCumsum, Untilize) | 0.077 | 0.114 | - | whole tensor |
| qk_norm | 0.060 | 0.114 | 0.060 | per segment |
| kv_write (typecast + UpdatePaddedKvCache, incl. index_k) | 0.021 | 0.038 | 0.014 | per segment |
| **misc total** | **0.893** | **1.733** | **0.749** | |

**What packing adds** (sparse misc, chip mean):

| W | single | same request as B segments (no h=0) | packed with an h=0 segment | per-segment cost | h=0 wait (routing_setup AllGather) |
|---:|---:|---:|---:|---:|---:|
| 4096 (B=2) | 0.893 | 1.201 | 1.817 | **+0.31 ms** (segment glue 0.240) | 0.626 against 0.011 ms |
| 8192 (B=4) | 1.733 | 2.339 | 2.980 | **+0.61 ms** (segment glue 0.481) | 0.658 against 0.013 ms |

* **Per-segment op counts:** 16 (B=2) and 32 (B=4) RoPE ops per sparse layer, against 8 for a single request, and 12 / 24 KV-write ops against 6.
* **Costs follow segments, not requests.** One slot split into 2 segments costs the same as 2 slots.
* **The packed-only glue** (`seg_slice_x`, `seg_slice_qkv`, `seg_concat_out`) is the largest single item: 0.140 + 0.054 + 0.047 ms at B=2.
* **Dense misc grows by +0.17 ms (B=2) and +0.36 ms (B=4).**

These are the inputs for batched RoPE, KV-write and variable-segment work. The whole-tensor ops (norms, residuals,
routing_setup) need no change.

## 12. P2: router, TP collectives, depth ops (`p2_summary.md`)

* **Router and TP collectives are flat in depth and in packing.**
  * router: 0.187 / 0.349 ms (6%);
  * norm_ag: 1.00 / 1.96 ms mean (38%);
  * attn_rs: 0.48 / 0.92 ms (39-41%), rising to 0.54 / 1.02 ms at 549k (arrival waits).
  * Here eff = roof / mean with no floor, so these read 2-4 pts below section 1.
* **routing_setup is the h=0 wait:** 2.12 / 2.44 ms worst at h=0 against 0.09 / 0.15 ms at 139k.
* **Growth with history** (chip mean, h = 0 → 549k, per 100k tokens):

| op | W=4096 ms at 0 / 139k / 549k | per 100k | W=8192 ms at 0 / 139k / 549k | per 100k | eff at 549k |
|---|---|---:|---|---:|---:|
| ag_kv | 0.12 / 0.47 / 1.70 | +0.29 | 0.38 / 0.64 / 1.89 | +0.27 | 40-44% |
| ag_idx | 0.02 / 0.22 / 0.84 | +0.15 | 0.04 / 0.23 / 0.84 | +0.15 | about 45% with the bf8 roofline (the file shows 85% against bf16) |
| indexer | 0.14 / 0.57 / 1.89 | +0.32 | 0.30 / 1.19 / 3.80 | +0.64 | 2.3% |
| sparse | 4.18 / 2.95 / 2.89 | flat after h=0 | 9.81 / 6.30 / 6.28 | flat after h=0 | 3.6-3.9% |
| sparse layer total | 15.03 / 13.81 / 17.01 | +0.36 | 29.25 / 25.72 / 30.54 | +0.24 | |

At 549k the depth ops (ag_kv + ag_idx + indexer) take 26% (W=4096) and 22% (W=8192) of the sparse layer's worst
chip. The indexer is the fastest-growing op.

## Methods, conditions, caveats

**Harnesses.**

* **P0-A zone profiles.** `run_prefill_profile.sh`, `STAGES=4 STAGE=0 FABRIC=1d`, `M3_PROFILE_LEVEL=2`, `PROFILE_SKIP_COMPILE=1`.
  * There is a warm point: forward #2 repeated 3x. The profiled forward is the 7th execution of the same programs.
  * At depth there is a real prefix with `PROFILE_PREFIX_QUIET=1`. Captures are 3.0-3.5 GB single and 5.3-7.4 GB packed.
  * Each run has one row in `runs.csv`, with the SHA and env.
  * The W=4096 h=548864 runs first failed on the tracy 32K source-location limit. They were rerun with the real-time profiler lanes muted.
* **P0-A2 (4,2).** `tools/profile_4x2.py` carves `create_submeshes(MeshShape(4,2))[0]` (rows 0-3, cols 0-1) out of the opened galaxy. It monkeypatches a 2-head KV cache and a per-head MSA cache-read gather.
  * The [4,2] weight cache for layers 0-6 was built in 231 s (20 GB, on weka).
  * The packed W=8192 run needed a 3200-program profiler buffer.
  * The earlier 4x2 prior art was Pavlo's harness `/data/philei/scripts/m3_sptp/profile_prefill_sptp.py`, which ran only layers 0 and 3.
* **Benches** (`bench/README_bench.md`). One process per grid, run on `create_submeshes(MeshShape(2,4))[0]` with the device profiler and no tracy.
  * Each point is 2 warm-up calls and then 5 timed calls; each chip takes the median.
  * Every device process was preceded by `tt-smi -glx_reset` with `env -u TT_VISIBLE_DEVICES`, under a 20 min timeout and a 5 min no-output watchdog.
* **Sim.** `tools/sim_core.js` from `philei/m3-traffic-sim` @ 446896d, driven by `tools/run_goodput.js`.
  * 4 galaxies, `opEff 0`, decode at 180 tok/s, 1800 s window.
  * Traffic: `/data/philei/m3_traffic_sim/data`.
  * `moeMult` is kept. `pipe.ringC` is scaled by our zone ring_c (0.3165 / 0.2961); it was not refit.
  * There is no `node` on PATH on this host; the tools use the Node that ships with the VS Code server (`NODE` env).
* **Roofline.** `tools/roofline_ops.js --segments W:139264 --idx bf8`. The experts roofline carries imbalance 1.2 with Tr = W.

**Caveats.**

* **Statistics.** Section 1 eff is mean chip, the same as Pavlo's. Share is worst chip. The worst-chip per-op times do not sum to the layer, because the waits sit on different chips.
* **The sim models imbalance only as the fixed 1.2.** The measured 1.5-2.1 sits inside the combine, moe_reduce and misc means.
* **The sim clips at target.** With `opEff 0` it uses min(eff, target), so qkv (W=8192) and experts have no effect beyond target.
* **Every [4,2] number comes from one single-stage profile, prose only,** with harness KV patches and 1D fabric. The native gather is an estimate: measured ag_kv minus the measured copies. There is no [4,2] pipeline measurement.
* **Pipeline hot streams use a synthetic prefix**, so the history KV is never written, and ringC was not refit to the new runner data.
* **The effs are taken at h=139k.** The sim applies them at every depth, but the depth ops do not scale with depth the way their rooflines do.

**Where source files disagree, and which one we use:**

1. **The experts fit reading** (section 7). `moe_chain_summary.md` and `pavlo_table_ours.md` say the hybrid is additive and name the overlap as the fix. We use `experts_fit.txt` directly: ND fits max (R² 0.998 against 0.971); the hybrid margin is small; and the two paths take equal time at the operating point.
2. **The ag_idx roofline.** `per_op.csv` `roof_ms`, `p2_summary.md` and `compare_4x2_vs_2x4.txt` use a bf16 index_k roofline, which gives ag_idx at 77-85%. `pavlo_table_ours.md` and the CAL.effs use bf8, which gives 47-54%. We use bf8, because the cache is `BFLOAT8_B`.
3. **eff definitions differ between files.** `p2_summary.md` uses roof / mean with no floor, pooling prose and code. `compare_4x2_vs_2x4.txt` uses roof / worst. Section 1 uses Pavlo's form, roof / (mean − floor).
4. **experts eff.** It is 78.3% / 74.8% in the prose-only table and 76.9% / 74.5% in the pooled CAL set. Both are correct for their pooling.
5. **The [4,2] cache build time.** `recipe_4x2.md` estimates 20-30 min for layers 0-6 (about 4 min per sparse layer) and about 57 x 4 min for all 60 layers. `kv_4x2_status.md` measured 231 s (about 45 s per sparse layer), so about 45 min for 60 layers. We use the measured value.
6. **Stale ringC statement.** `goodput_results.md` says no deep-context pipeline data exists to fit ringC. It was written before the `d_w*_sync1` hot-139k sessions, which now exist but were not used for a refit.
7. **The async-handoff hang count.** `pipeline_overheads.txt` says "reproduced once". That means one hang plus one reproduction: 2 hangs in total (`runs.csv`: both `STATUS=HANG`).

## Deliverable files

| file | contents |
|---|---|
| `REPORT_ops.md` | this report |
| `runs.csv` | append-only log of every device run: SHA, env, cmd, status, log path |
| `p0a_runs.csv`, `p4x2_runs.csv`, `p1f_runs.csv` | per-block run lists for P0-A, P0-A2 and P1-F |
| `per_op.csv` | [2,4] per-op zone times (worst / mean / min, roof, eff, share), layers 0-6, the full W x h x input grid plus packed |
| `per_op_4x2.csv` | the same for (4,2), prose plus packed, including the ag_kv head-copy sub-zones |
| `pavlo_table_ours.md` | Pavlo's table at our conditions, packed tables, the ranked list, the calibration diff, and CAL.effs['2x4'] |
| `pavlo_table_ours_4x2.md` | the [4,2] table with Pavlo's [4,2] column, [2,4] side by side, shares and worst-ms ratios |
| `pavlo_reference.md` | Pavlo's [2,4] / [4,2] tables, CAL.effs, pipe fit, goodput reproduction, and the "near-term" set search |
| `compare_4x2_vs_2x4.txt` | (4,2) against (2,4) per op and per layer, worst chip, all grid points |
| `kv_4x2_status.md` | (4,2) KV PCC, what breaks at TP=2, the sharding-change estimate, the head-copy cost |
| `recipe_4x2.md` | how to carve and run (4,2), and the runner pieces that are missing |
| `goodput_results.md` | CAL.effs['4x2'], 16 x [2,4] against 16 x [4,2] goodput, goodput per fix, the decision rule |
| `moe_chain_summary.md` | P0-B: chain per chip, load skew, the wait attribution, experts and moe_reduce analysis |
| `load_skew.csv` | expert load per chip, max/mean, and top-expert share per case and layer |
| `load_skew_join.csv` | load joined with experts zone time and the bench model (imbalance against kernel) |
| `experts_fit.txt` (= `bench/experts_fit.txt`) | sum and max fits for the hybrid and ND expert paths |
| `bench/experts.csv` | routed-expert bench grid |
| `moe_reduce_bench.csv` (= `bench/moe_reduce.csv`) | post_combine_reduce, RS, and fused + RS at 1k / 2k / 4k tokens per chip |
| `msa_summary.md` | P0-C: MSA op scaling, the cause of sparse_sdpa's cost, Anomaly B (h=0) |
| `msa_bench.csv` (= `bench/msa.csv`) | MSA op bench grid, rows x kv_len |
| `bench/README_bench.md` | bench method, inputs and op structure from the program factories |
| `pipeline_overheads.txt` | P1-D per-stage compute, blocking send, hop, host gap; the async A/B hang; KV read-back |
| `dense_scan_check.txt` | P1-E 64k against 1M slot capacity |
| `misc_breakdown.txt`, `misc_breakdown.csv` | P1-F misc per op, count, ms, and per-segment / whole-tensor tag |
| `p2_summary.md` | router, TP collectives and depth ops against history |
| `gate/gate_merge_L04_compare.txt` | post-merge packed-path gate, layers 0-4, worst PCC 0.99970 |
| `tools/roofline_ops.js` | roofline per op (the sim's `roofTok` / `roofSeg`) |
| `tools/cal_effs_ours_*.json` | our CAL.effs sets for [2,4] and [4,2], per W, including prose-only and native-gather variants |
| `goodput/*.json`, `goodput/fix/` | raw sim sweeps and per-fix effs files |
| `bench/`, `profiles/`, `pipeline/`, `skew/`, `logs/` | raw bench, zone, runner, load-stat and log outputs |

Not checked in: the raw tracy captures (`profiler_tmp/`, ~89 GB), the gate KV dumps (`gate/kv_*`, 3.8 GB) and the
code-corpus token file (`inputs/code_m3/metadata.json`, rebuilt by `tools/make_code_corpus.py`).
