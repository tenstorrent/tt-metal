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

## Reality note
Each remaining iteration is deep fused-op code surgery + 10-min reprofile + PCC verify. This is the
long autonomous grind the PLAN is built for; it is not flag-flipping. Batches 8/16/32 repeat this
loop (decode M grows with batch → matmul util improves, so SLOW-matmul pressure eases at higher batch;
TM structural overhead persists).
