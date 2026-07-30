# DiffusionGemma decode-throughput campaign — ledger

Status: provenance-only for every absolute throughput number below. The token-gather MoE that produced them was DELETED 2026-07-29, so nothing here is a current result; the levers, refutations and defect findings are what survive.
Owns: the token-gather MoE landed result, the sequential-vs-batched commit defect, the landed-lever commit provenance, and the self-conditioning selected default.
See also: [refuted list](../REFUTED.md) · [stage hub](README.md) · [roadmap arithmetic](path_to_100tps.md)

Absorbs `perf_campaign_worklog.md` and `landed_levers_47465_comment.md` (both deleted 2026-07-30). Metric definitions: `ms_per_block = steps_per_block × ms_per_denoise_step + commit_ms`; `tokens_per_block_per_s = 256 / (ms_per_block / 1000)`. Never `1000/mean_tpot_ms`.

## Authority matrix (2026-07-17), still binding

Pure prefill, selected fixed-K=48 output-block throughput, and live vLLM are **three separate authorities**. Never substitute raw frozen-prefix, early-halt or standalone-session rows for current live-serving throughput. Prefill authority: [stage hub](README.md#prefill-authority). Live serving: [vllm_integration](../vllm_integration/README.md). The 18.844 tok/s fixed-K=48 row was measured at `0472860c40c` on the deleted token-gather path and has never been rerun at HEAD.

## Landed levers and their commits (provenance)

| lever | effect as measured then | commits |
|---|---|---|
| Sparse token-gather MoE | MoE ~13× (dense 137.6 → ~10.5 ms/layer) | `e98fabaaff4` (lever-A GO, 12.5×, PCC 0.9997) · `1d1ccd93a8b` (land true-sparse, 9.3× traced step) |
| OPT-004 matmul geometry | MoE ~3.47× vs auto (PCC 0.99967) | `014c47177f7` (opt-in) · `9c5c999fb80` (default ON) |
| Terminal trim | argmax 1240 → 14.4 ms/op, 2×/step | `474713ec259` (dedup argmax) · `a39e8d63c1b` (ROW_MAJOR argmax + trace-safe terminal) |
| Traced denoise loop | 2.72× traced vs eager; @48 = 17.92 t/s | `d25626f2636` (wired into serving, bit-exact 58.29 t/s @12) · `35e70fd1225` (@48 = 17.92) |
| Batched commit as default | 2.54× on the block; commit stops dominating | `3d71dee8a97` |

Cumulative claim of that era: model-faithful @48 went from ~1.08 t/s to ~18 t/s. **Four of the five
levers describe code that has since been deleted**, and the "precision-neutral in-repo @48 ceiling
~18 t/s" was superseded twice — see the [refuted list](../REFUTED.md).

## Dense-128 provenance (the wall the campaign started at)

The experts-only MoE forward was **137.60 ms/layer**, ~99% of the ~4176 ms 30-layer traced step, and sat ~65–85× above BOTH its ~1.6 ms weight roofline and its ~2 ms compute roofline — i.e. data-movement / op-overhead bound, dominated by the ~87 ms expert-major `Permute`. Step decomposition and the weight-byte model: [work_log.md](work_log.md).

## Token-gather MoE — landed result, then deleted

On-device GShard capacity dispatch (`topk → scatter → cumsum → gather → col math → scatter`, then gather = `disp^T @ x`, batched experts, combine = `comb @ down`, all-reduce) measured **10.54 ms/layer at C=32** vs dense 137.07 (**13.0×**) with dispatch-build only **1.87 ms** and MoE **PCC 0.99969**; gather/scatter overhead was ~2 ms, not a washout. Implementation fact that outlived the path: **dispatch index tensors must be UINT32** because ttnn gather/scatter reject INT32. Where the 13× came from: token-expert products per layer fell from 8 chunks × 128 × 32 = **32,768** to 128 × 32 = **4,096**, and the ~87 ms/layer expert-major `Permute` was eliminated because experts became the leading batch dim. The path was deleted 2026-07-29 — reason and cost analysis in [winter borrow](winter_borrow_20260727.md) and the [refuted list](../REFUTED.md).

## Op-topology audit of the traced step (2026-07-08, tuned era)

Traced denoise step **233.4 ms** vs eager **720.8 ms** (3.1×; dispatch overhead is 68% of the eager step); commit eager 129.0 ms/token; prefill TTFT for an 18-token prompt 607.9 ms. Shares of the traced 233 ms step: Matmul 35% (~82 ms), layout/glue 28% (~65 ms), elementwise/reduce 22% (~51 ms), LayerNorm 6%, TP collectives 4%, ArgMax 4%. The `Permute` cumsum artifact is **gone** (1.8%), so the old #47465 `SparseMatmul+Permute` breakdown is obsolete. Roofline framing: each denoise step re-reads all resident weights (13.1 GiB/chip, 88.6% MoE experts) over the full 256 canvas, so weight traffic — not incremental KV — sets the floor.

Per-component profile (`prof_step_breakdown.py`, synchronized): the 30-layer forward is ~433 ms = 89% of the ~484 ms step; terminal 28.99 ms (6%), soft-embedding 16.04 ms (3%), LM head 4.36 ms, self-cond gated MLP 1.69 ms. **The step is backbone-forward-bound** (per-layer attention + 2 TP all-reduces), not MoE, terminal or serving glue.

## The commit: a correctness fix, not just a perf lever

- **LANDED, bit-exact by construction:** the commit's LM-head skip (`skip_lm_head` in
  `commit_decode_forward` / `_commit_model_forward`) removes a discarded 2816×262144 LM head + final
  norm per token — ~83 ms/block, layer-count independent.
- **DECISIVE:** the sequential decode-commit MoE is genuinely **DEFECTIVE**. The torch oracle
  `probe_moe_vs_torch.py`, on the bit-exact layer-0 input, gives PCC(torch, batched) = **0.9936** vs
  PCC(torch, sequential) = **0.1542** (a later re-run: 0.856 vs 0.579). So the batched commit was a
  correctness fix. `_commit_experts_decode_forward` is a near-verbatim copy of the shared gemma4
  decode expert kernel, which the batched path deliberately avoids.
- **RETRACTED misdiagnosis (one line):** "batched-commit attention is buggy" was wrong —
  attention-only KV PCC is 0.9977 at 4 layers, and the 30-layer decay to 0.494 is a bf16
  prefill-vs-decode kernel difference amplified by depth.
- **MEASUREMENT TRAP:** `verify_commit_batching.py` (batched vs sequential KV PCC at 30 layers,
  0.997 bar) is an **INVALID gate** — it measures the correct path against the defective reference,
  and no two non-bit-identical commit implementations can meet 0.997 at 30 layers because bf16
  differences chaos-amplify through residuals. Details: [commit_batching.md](commit_batching.md).

## Verdict lines carried forward

- **bfp8 experts: REJECTED** on fidelity (committed clean-argmax agreement 0.227) despite DRAM
  13.268 → 7.830 GiB/chip (−41%) and @48 18.18 → 19.83 t/s (+9.1%) — only ~6–9%, because the step is
  matmul launch/overhead limited, not weight bound. Full sweep: dg-07 / #47475, [datatype_sweep](../datatype_sweep/README.md).
- **Early halt:** mechanism correct and eager-faithful (per-step device `(mean_entropy, mismatch)`
  agree with eager records to 1.2e-5 and exactly 0; no-halt commits byte-identical to fixed-48
  `a9f0d18709b07d1e`). Overhead is 5.87 ms/step (scheme A, 48 syncs), ~2% of the block, so break-even
  is ~47 of 48 steps and **any early halt wins**. Firing status: [early_halt.md](early_halt.md).
  > **OPEN CONTRADICTION (unexplained):** the eager halt oracle (`probe_halt_gap.py`, 30L, 3 blocks)
  > on the token-gather MoE ran every block to the full 48 steps — the stability gate fired (14–18
  > argmax-stable steps) but mean entropy floored at ~0.14–0.51 nats, 30–100× the 0.005 threshold —
  > and eager adaptive @48 ran `steps=[48,48]`, `halted=[False,False]`; under the concat MoE the halt
  > fires 19/19 in 8–27 steps at entropy ~6e-4 (`tt/concat_moe.py`), halts of `[9,17,2]`/48 and
  > K=10–43 are recorded elsewhere, and real blocks were separately observed halting adaptively at
  > **18–38 steps** (recorded `[27]`, `[36,18]`, `[38]`), giving the 1.3–2.3 t/s of the dense era.
  > Not explained.
- **Full-canvas RMSNorm micro-evidence:** per-norm 1.32 → 0.134 ms (9.8×), traced e2e @48
  17.855 → 20.676 t/s (+15.8%) and @12 +23.3%, worth ~41 ms/step; it survives trace because it
  removes real `Slice`/`Concat`/norm-launch ops. The flip gate that once blocked it is void — see
  [stage hub](README.md) and [l1_residency.md](l1_residency.md).
- **`DG_MOE_L1`, multi-step trace batching, the compact ragged MoE, the two-stage vocab reduction,
  the norm `ttnn.split`, the GQA-fallback claim, OPT-004 block-size tuning** — all refuted; one line
  each in the [refuted list](../REFUTED.md).
- **OPT-004 provenance on the deleted path:** full `sparse_experts_forward` **10.083 → 2.892 ms
  (3.49×) at PCC 0.99964**, and the dominant gate/up matmul **4.176 → 0.593 ms (7.05×) at PCC
  0.99986** — the win came from a `program_config` existing at all, not from block size. A separate
  reading of the same lever is in [opt004_matmul_geometry.md](opt004_matmul_geometry.md).
- **FUSED2 dispatch provenance:** bit-identical end-to-end (identical `committed_sha`) at +2.3% @48 /
  +5.2% @12, flipped default ON, then deleted with the token-gather path.
- **Compact ragged MoE, repaired exact mode:** primary gate/up K-block 8 plus down K-block 2 with the
  `dense_compat` combine restored identical SHAs and gave +4.8% @48 / +3.1% @12. The flag family was
  deleted 2026-07-28.

## Self-conditioning selected default (2026-07-10) — provenance

Prechunked embedding (32 persistent 8K-row tensors, removing 32 device `Slice` ops per denoise step,
**+3.03% @48**: 14.0971 → 13.6817 s, 18.160 → 18.711 t/s, exact established SHAs, 48/48 exact
per-step decisions in argmax and chunked-Gumbel; `DG_SELFCOND_PRECHUNK_EMBED=0` is the diagnostic
opt-out) plus the logits-L1 chain (soft embedding **18.213 → 16.038 ms, −11.94%**). Final unset-default
reproduction: **13.5849 s / 18.844 t/s @48**, 4.3122 s / 59.366 t/s @12, i.e. **257.575 ms/warmed
traced step**. Detail: [selfcond_prechunk.md](selfcond_prechunk.md), `selfcond_logits_l1_e2e.json`.

**HONEST LIMITATION kept:** that default is +0.71% on the block but complete three-block generation
REGRESSED 153.3410 → 153.9791 s (**+0.42%**), so no full-generation win is claimed.

> **OPEN CONTRADICTION (unexplained):** three independent-process controls improved the median
> 13.6161 → 13.5111 s while a same-model sequential A/B regressed 13.6456 → 13.7841 s, and one @12
> explicit-candidate sample also regressed. Both readings were deliberately retained. Not explained.

**OPEN capacity limit:** full-depth **traced** 256K is not composable — with no reservation the trace
overlaps the DRAM high-water mark, while 176–512 MiB reservations leave no contiguous 128 MiB
token-entropy temporary. The eager 256K capability is preserved; the limit is recorded in
`doc/context_contract.json`.

## Measurement traps from this campaign

- The whole-run tracy `GAP_FRACTION=81%` is a **device-profiler artifact**, not real host gap: clean
  wall-clock shows only 27.76 ms/step of host overhead.
- The Tracy device profiler dropped device rows, so `SparseMatmulDeviceOperation` read **0 ms** in the
  denoise region while being 16.27 ms in prefill — treat absolute op ms from that capture as **lower bounds**.
- `serving_smoke` breaks on `session.finished` after block 0, so by default it only measures the cold
  block that pays prefill plus trace capture. Steady state needs `stop_token_ids=[]` and `mean(blocks[1:])`.
- An apparent "~54 minute hang" or ">7 minute compile" was a **COLD JIT cache**, not a deadlock — with
  the disk cache warm the same probe reached trace capture in ~4 minutes.
- Isolated-lever projections summed to ~150 ms/step but the same-session serving A/B measured only
  ~40 ms/step, because the eager serving step was host-dispatch-bound and masked compute-time wins.
- `bench_lever_e2e`'s "@48" rows actually halted at ~9 steps because early halt was default-on, which
  produced a bogus "~50 t/s @48 and a 3.5× vLLM plumbing overhead" reading. **There is no vLLM
  plumbing overhead**; full-48 was ~11–12 t/s.
- The July-15 per-component attribution used `capacity=32` in an isolated `moe_layer` call while
  production used `capacity=canvas_length=256`, so "MoE experts 2.63 ms/layer" was wrong — at the
  effective capacity it is **9.11–9.21 ms/layer**, and C=32 drops 41–84% of routes on concentrated
  real routing.
- The **plain-watcher `ACTIVE_ETH` kernel-config-buffer overflow** is a known watcher+fabric
  limitation, not a model defect — run watcher with `TT_METAL_WATCHER_DISABLE_ETH=1`.
- **METHODOLOGY (load-bearing):** the denoise step count is a **MODEL parameter, not a free perf
  knob**. The HF reference halts on `StableAndConfidentStoppingCriteria` with
  `stable_steps_to_halt=1`, `entropy_stop_threshold=0.005` nats and cap `max_denoising_steps=48`, all
  verified against the checkpoint `generation_config`. The @12/@8/@6/@5/@4/@3 descent
  (58.2 / 70.1 / 85.6 / 94–95 / 104.3 / 119.4 t/s) and the fit `block(K) ≈ 0.239·K + 1.584 s` at 30L
  traced are a per-step **COST** characterization, never a throughput the model delivers.
- Device grid is **11×10 = 110 cores**, not the 13×10 = 130 the design doc assumed.

## Retired harnesses

`sweep_at48.py`, `sweep_serving.py`, `bench_loop_readback.py`, `probe_traced_denoise_loop.py`,
`probe_singlestep_traced.py`, `probe_traced_serving.py`, `probe_selfcond_race.py`,
`verify_terminal_dedup.py`, `verify_opt004_fullmoe.py`, `bench_opt004_matmul_geometry.py` and
`verify_sparse_moe.py` no longer exist in the tree. The measurements they produced stay on record here.
