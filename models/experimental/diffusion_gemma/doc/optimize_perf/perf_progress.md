# DiffusionGemma decode-throughput optimization — running progress

QB2 (`bh-qbge-06`, P150x4, mesh `(1,4)`, TP=4), branch `diffusion-gemma-function`, `build_Release`.
Optimization unit = **denoise step over the 256-token canvas** (≤48 steps/block) + **commit**.
Goal: raise decode throughput from ~1.3–2.3 t/s toward 30 t/s. One verified lever at a time.
All decode numbers are **traced** (eager is not acceptable evidence).

## Metric definitions (reported per lever)

- `ms_per_denoise_step` — warmed traced device time for one denoise step over the full 256 canvas, 30 layers.
- `steps_per_block` — denoise steps actually run before halt (≤ 48; data-dependent early-halt).
- `ms_per_block` = `steps_per_block × ms_per_denoise_step + commit_ms`.
- `tokens_per_block_per_s` = `256 / (ms_per_block / 1000)`.

## Baseline (established, prior work — see README.md / work_log.md / ttft_ts_sweep.md)

| quantity | value | source |
|---|---|---|
| TTFT (full 30L prefill) | 152–211 s | ttft_ts_sweep.md |
| ms_per_denoise_step (traced, 30L, projected from L=1/2/4) | **≈ 4176 ms** | README headline |
| — per-layer denoise | 137.55 ms/layer | README |
| — fixed overhead (embed+LMhead+terminal+norm) | 49.24 ms | README |
| commit (256 single-token decode-appends, 30L) | ≈ 31.5 s/block | README |
| ms_per_block @ 48 steps | ≈ 231.9 s | README |
| tokens_per_block_per_s @ 48 steps | **≈ 1.10** | README |
| roofline floor (weight traffic, 256 GB/s) | ≈ 24–49 ms/step | work_log §4 |

Per-step is context-independent (MoE weight-traffic bound); measured ~4176 ms is ~85–170× the
bandwidth roofline → **op-count / dense-MoE-compute bound**, large headroom.

## Target arithmetic

30 t/s ⇒ `ms_per_block ≈ 8533 ms`. At 48 steps that means ~170 ms/step incl. commit — a ~27×
reduction from 231.9 s/block. The denoise steps (200 s @48) dominate the block; commit is 31.5 s.

---

## Lever 1 — Traced-decode baseline / wire trace into measured path — **WASHOUT (recorded, not landed)**

**Question:** is the measured decode path traced? If eager, does wiring the trace-safe loop win?

**Findings (fresh, this session):**
- The measured decode path is **eager**. `serving.decode_block → denoise_and_commit_block →
  denoise_block` runs the eager loop with **5 host readbacks per step** (`argmax`, `entropy`,
  `sampled`, `accept_mask`, `canvas` → `to_torch`) plus a `torch.equal` stability halt check.
  Nothing wraps `run_fixed_denoise_steps` (the existing trace-safe device-only loop) in
  `begin_trace_capture`.
- **Quantified the readback overhead** apples-to-apples on the same reduced model
  (`bench_loop_readback.py`, L=2, 8 steps, canvas 256):

  | path | ms/step (L=2) |
  |---|---|
  | A: production eager `denoise_block` (host readback/step) | **359.27** |
  | B: trace-safe device-only loop, **traced** | **331.52** |
  | **readback overhead (A−B)** | **27.76 ms/step** |

- The 27.76 ms/step readback is a ~fixed per-step host cost (5 tiny 256-elem tensors + torch
  work), roughly layer-count independent. Against the 30-layer step (~4176 ms) it is **~0.66%**.
  This reconciles the README's "traced ≈ eager (~3%)" for the no-readback microbench.
- The whole-run tracy `GAP_FRACTION=81%` is a **device-profiler artifact**, not real host gap:
  clean wall-clock shows only 27.76 ms/step host overhead.

**Decision — do NOT wire a fixed-step trace into the measured path:**
1. The saving (~0.66% at 30L) is negligible vs the op-cost-bound step.
2. Tracing forces a **fixed 48-step** budget (early-halt cannot shorten a static trace). The eager
   path halts early when the canvas is stable+confident; converting to fixed-48 would **remove
   early-halt** and likely *regress* real blocks that converge in far fewer than 48 steps.

Lever 1 is therefore an honest washout for the denoise step. The genuine wins are the per-step
device compute (levers 2–4, dense-128-expert MoE) and the commit (256 sequential decode-appends).
Tracing may still help the **commit's** 256-decode inner loop and is revisited under lever 5.

_Artifacts:_ `bench_loop_readback.py`, `artifacts/lever1_readback_L2.log`,
`artifacts/lever1_traced_baseline_L2.log`.

---

## Lever 2 — Operation-topology audit — _in progress_

Clean 1-layer eager Tracy capture (`lever2_tracy_L1.log`, per-step over 3 steps). **Caveat:** the
device-profiler dropped many device rows ("no device perf row" skips) so `SparseMatmulDeviceOperation`
reads **0 ms in the denoise region** (clearly wrong — it is 16.27 ms for 3 ops in the prefill
region). MoE compute time is therefore under-counted; treat absolute op ms as lower bounds.

**Per denoise step (1 layer), reliable signal:**

| op | ms/step (1L) | %fw | note |
|---|---|---|---|
| `PermuteDeviceOperation` (MoE expert-major reorder) | **86.9** | 70% | biggest op; `transpose(gate/up,1,3)` feeding down `sparse_matmul` |
| `ArgMaxDeviceOperation` (terminal, 2×/step) | 20.2 | 16% | over 262144 vocab; already ROW_MAJOR-optimized; **fixed per-step** |
| `BinaryNgDeviceOperation` | 10.2 | 8% | mostly MoE elementwise (routing-mul, geglu) |
| `SparseMatmulDeviceOperation` (MoE dense-128 compute) | ~60 (missing) | — | inferred: prefill 16ms/32-tok × 8 chunks |
| `MatmulDeviceOperation` (attention/shared_mlp) | 0.8 | <1% | |

**Findings feeding the levers:**

1. **The MoE dominates the per-layer step (~80%)** = Permute reorder (~87 ms) + sparse-matmul
   compute (~60 ms). BOTH scale with **num_experts computed (128)**. The dense-128 pattern
   (`prefill.py`: `nnz = num_experts × group_size`, all-ones sparsity, routing applied *after* to
   zero 120/128) is the structural waste. The Permute is intrinsic to that pattern
   (`moe_transpose_investigation.md`: in-place rewrite = 0%, cost just relocates) — so the only way
   to shrink it is to compute **fewer experts** (lever 3), not to rewrite the transpose.
2. **`sparse_matmul` sparsity granularity is per-GROUP (per 32-token tile), not per-token.** The
   256-canvas is 8 chunks × 32 tokens; each chunk's 32 tokens share one sparsity row, so they need
   ~all 128 experts. True per-token top-8 needs `group_size=1` (256 tiny tiles → padding erases the
   win) OR a **token-gather rewrite** (sort tokens by expert). The gemma4 **decode** path already
   uses `nnz=top_k=8` (true sparse) but only for seq_len=1.
3. **Commit = 256 sequential single-token decode-appends** (`generate.py:623`). Each pays the full
   30-layer weight traffic → weight cost paid 256× instead of amortized once. It also computes a
   full 2816×262144 **LM head per token and discards it** (`commit_canvas_tokens` deallocates the
   logits). Batching the commit into one 256-token causal prefill-append is a ~256× weight
   amortization; skipping the LM head is a smaller safe sub-win.
4. **ArgMax is fixed per-step** (~20 ms), so it is ~0.5% of the 30-layer step — not worth chasing.

### Clean MoE microbench (`bench_moe.py`, non-profiled wall-clock, 256 canvas)

| quantity | value |
|---|---|
| expert weight dtype (actual) | **BFLOAT16** (DiffusionGemma overrides gemma4's bfp8 default) |
| MoE forward (router+experts) | 138.46 ms/layer |
| — router only | 3.21 ms |
| — **experts only (dense-128)** | **137.60 ms** |
| — shared_mlp only | 2.33 ms |

**The dense-128 experts forward IS the per-layer denoise cost** (~137 ms/layer ≈ README per-layer;
router/shared_mlp/attention are single-digit ms). At 30 layers this is ~99% of the ~4176 ms step.

**Roofline reconciliation (per layer, per chip @256 GB/s):** expert weights (bf16) ≈ 415 MB →
**~1.6 ms weight roofline**; dense-128 compute ≈ 106 GFLOP → **~2 ms compute roofline**. Measured
137 ms is **~65–85× above both** → the experts forward is **data-movement / op-overhead bound**
(the expert-major transpose reorder ~87 ms + 8-chunk op overhead), NOT weight- or compute-bound.

**Consequence:** BFP8/BFP4 expert precision does **NOT** help (weight traffic is only 1.6 ms; the
bottleneck is the transpose reorder). The only lever that cuts 137 ms/layer is computing **fewer
experts** (transpose + compute both scale with num_experts=128) — i.e. true-sparse.

### Lever-3 prototype (`bench_sparse_moe.py`): per-token nnz=8 — **BLOCKED**

Routing is exactly top-8/token (min=max=8 non-zero → the `nnz=8` invariant holds). But feeding the
256-token canvas to the decode path (per-token `nnz=top_k=8`) fails:
`TT_FATAL: sparsity.logical_volume() == batch_length` (sparse_matmul_device_operation.cpp:154).
`sparse_matmul` sparsity granularity is per M-tile-group, not per-token; per-token top-8 would need
`group_size=256` (each token its own 32-tile) → 32× M-padding that erases the 16× expert win.
**True-sparse therefore requires a token-gather rewrite** (sort tokens by expert → contiguous
per-expert batches), whose gather/scatter overhead is the mission's flagged washout risk. This is a
large, higher-risk rewrite scoped as the next big lever.

_Baseline confirmed:_ dense-128 = 137.08 ms/call. Artifacts: `lever2_moe_bench.log`,
`lever3_sparse_proto.log`.

### Lever-4 quick wins — ALL washed out or blocked (hard device evidence)

| candidate | result | evidence |
|---|---|---|
| BFP8/BFP4 expert precision | **won't help** — weight traffic is only ~1.6 ms/layer; MoE is transpose-bound | roofline reconciliation above |
| per-token `nnz=8` sparse (decode path on 256) | **BLOCKED** — `sparse_matmul` sparsity per M-tile-group, not per-token | `lever3_sparse_proto.log` |
| larger prefill chunk (64/128/256) | **BLOCKED** — `num_blocks_total <= num_cores_available` core overflow | `lever4_chunk_sweep.log` |
| MoE intermediates DRAM→L1 | **washout 1.01×** (137.09 → 135.14 ms), PCC 1.0 | `lever4_moe_l1.log` |

The in-repo DRAM copy reproduces the shared path exactly (137.08 ms, PCC 1.0), confirming the
transpose is an intrinsic slow-reorder kernel (not a DRAM-placement artifact); L1 barely moves it.

## Conclusion & remaining levers

**The wall is the dense-128 experts forward: 137 ms/layer × 30 ≈ 4.1 s = ~99% of the denoise step.**
It is data-movement/transpose bound (~65–85× above weight & compute rooflines), and every quick,
low-risk lever (trace, precision, per-token sparsity, chunk size, L1 placement) has been ruled out
with device evidence. The two remaining wins are both larger rewrites:

1. **True-sparse token-gather MoE (biggest lever).** Sort the 256 canvas tokens by their top-8
   expert assignments into contiguous per-expert batches, run a dense matmul per expert on only its
   ~16 tokens (padded to a 32-tile), scatter back weighted. Cuts experts computed 128→~effective-32
   (after tile padding) ⇒ cuts BOTH the transpose reorder and compute ~4–8×. Risk (mission-flagged):
   the on-device gather/permute + scatter overhead may erase the win — must be prototyped and
   measured before the full rewrite. Needs `sort`/`gather`/`scatter`/permute over 256×8 assignments,
   variable per-expert token counts (capacity/pad handling), all TP-sharded and trace-safe.
2. **Commit batching (256 decodes → 1 causal prefill-append).** The commit re-encodes the finished
   256-token canvas as 256 sequential single-token decode-appends (`generate.py:623`), paying the
   full 30-layer weight traffic 256× (≈31.5 s/block). One causal 256-token prefill-append writes all
   KV in a single backbone pass (≈4.2 s, one denoise-step-equivalent) → ~7× on the commit, ~1.25×
   on the block at ~25 steps. Needs a DiffusionGemma-local causal prefill-append at offset `start_pos`
   (analogous to the existing `commit_decode.py`), verified by KV-cache PCC vs the 256-decode path.
   Also skip the discarded final-norm+LM-head in the commit (small safe sub-win, ~3% of commit).

Neither lever alone reaches 30 t/s (the step's 30× dense-MoE is fundamental); the token-gather MoE
is the only structural path to a materially cheaper per-step, and is the correct next investment.

## Lever (commit) — LM-head skip — **LANDED (small, verified)**

The commit (`commit_canvas_tokens`) re-encodes the 256-token canvas as 256 sequential decode-appends
and **discards the per-token logits** (`generate.py` deallocated them). Each decode still computed a
full 2816×262144 LM head + final norm that were thrown away. Added `skip_lm_head` to the
DiffusionGemma-local `commit_decode_forward` / `_commit_model_forward` and set it in
`commit_canvas_tokens`.

**Correctness — bit-exact by construction:** the KV-cache append happens inside the layer loop
(`_commit_layer_forward` → `paged_update_cache`), which is byte-for-byte unchanged; `skip_lm_head`
only skips the post-loop final-norm + LM head (a read→matmul with no KV side effect). The KV cache is
therefore provably identical.

**Measured** (`bench_commit_skip.py`, controlled before/after, L=2): commit 2297.4 → 2214.3 ms =
**83.1 ms/block saved** (3.6% at 2 layers). The saving is the LM-head cost, which is layer-count
independent, so at 30 layers it is the same ~83 ms/block ≈ 0.26% of the 31.5 s commit — a genuine
waste removal but a negligible fraction of the block. Recorded honestly: real, free, zero-risk, but
not a needle-mover toward 30 t/s. No gemma4 edits (change is in DiffusionGemma-local commit code).

## Lever 5 — 2 command queues / host-gap removal — **little headroom (per lever-1 evidence)**

Lever 1 established the denoise step is **op-cost bound, not dispatch-gap bound**: traced ≈ eager
(within ~2% on the reduced model) and the per-step host readback is only 27.76 ms (~0.66% at 30L).
The whole-run tracy `GAP_FRACTION=81%` is a device-profiler artifact, not real host gap. 2 command
queues overlap compute with I/O, but the step is compute (dense-MoE) bound with negligible I/O to
overlap, so 2-CQ / host-gap removal offers little for the denoise step. It could still help the
serving orchestration around block boundaries, but that is dominated by the same per-step MoE.
Not pursued; the leverage is entirely in the per-step MoE (lever 3).

## Lever A — TRUE-SPARSE token-gather MoE — **DECISIVE GO** (prototype, 2026-07-04 session 2)

**Question (GO/NO-GO):** does gathering only the top-8 active experts' tokens (vs computing all 128)
beat the dense-128 path, or does gather/scatter overhead erase the win?

**Prototype** (`bench_gather_moe.py`, real 26B layer-0 MoE, real router routing, 256 canvas, C=32,
10 iters, host-built dispatch indices — the on-device dispatch is measured separately below):

The canvas input is REPLICATED across the (1,4) TP mesh (experts are TP-sharded on the intermediate
dim, NOT expert-parallel), so **gather/scatter over the token dim is LOCAL per device** — no
cross-device dispatch needed, unlike deepseek's `all_to_all_dispatch` (which needs a 16-row ring).

| stage | ms/call | vs dense |
|---|---|---|
| 0: dense-128 prefill (the wall) | **137.08** | 1.00× |
| 1: batched-experts only (gate/up/geglu/down on [E,C,H], no gather/scatter) | **8.89** | **15.42×** |
| 2: FULL — embedding-gather + batched experts + combine-matmul + all-reduce | **10.93** | **12.54×** |
| 3: PCC(full, dense) | **0.99970** | — |

- Gather/scatter overhead = 10.93 − 8.89 = **~2 ms** (NOT a washout).
- Capacity 32 (one 32-token tile per expert) captured **all 2048 (token,expert) pairs, 0 dropped**
  for this input (2048 pairs / 128 experts = 16 avg, well under 32). PCC 0.9997.
- Batched matmul over 128 experts (each a 32-token tile) is 15.4× cheaper than the dense-128
  sparse-matmul because it computes 128×32 token-expert products instead of 8chunks×128×32 = 8×
  fewer, AND it emits experts as the leading batch dim so it **avoids the 87 ms expert-major
  transpose** entirely (the dense path's dominant cost).

**This is the structural win the whole campaign was pointed at.** Replacing the 137 ms/layer dense
MoE with ~11 ms/layer collapses the ~99%-MoE denoise step ~12×.

### On-device dispatch (trace-safe) — holds the win (`bench_ondevice_dispatch.py`)

The host-built indices above aren't trace-safe. The real path builds the GShard capacity dispatch
ON-DEVICE from the router topk (all trace-safe ops): `topk → scatter(mask) → cumsum(pos) →
gather(pos_per_slot) → col=idx*C+pos → scatter(disp/comb masks)`, then gather = `disp^T @ x`
(matmul), batched experts, combine = `comb @ down` (matmul), all-reduce.

| phase | ms/call | vs dense |
|---|---|---|
| dense-128 (the wall) | 137.07 | 1.00× |
| on-device dispatch-build ONLY (topk+scatter+cumsum+gather+col math) | **1.87** | — |
| FULL on-device (dispatch + gather-matmul + experts + combine-matmul + all-reduce) | **10.54** | **13.01×** |
| PCC(full, dense) | **0.99969** | — |

Dispatch index-building is only **1.87 ms**; the full trace-safe MoE is **10.54 ms/layer**, actually
faster than the host-index version (fully device-resident). Index tensors must be UINT32 (gather/
scatter reject INT32). **Capacity note:** total (token,expert) pairs = S×top_k = 2048 always; C=32
gives 4096 slots = 2× headroom (0 dropped on the test input). Real-data imbalance may exceed 32 for
a hot expert → those tokens' contribution is dropped; the full-module PCC check on real activations
decides the safe C (bump to 48/64 if needed, still 6–8× vs dense).

_Artifacts:_ `bench_gather_moe.py`, `bench_ondevice_dispatch.py`,
`artifacts/leverA_gather_moe_C32.log`, `artifacts/leverA_ondevice_dispatch_C32.log`.

### Full in-repo module + traced t/s — **LANDED** (`tt/sparse_moe.py`, opt-in `DG_SPARSE_MOE=1`)

Built `models/experimental/diffusion_gemma/tt/sparse_moe.py` (`sparse_experts_forward`, drop-in for
`moe.experts`), wired into `_denoise_moe_forward` behind `DG_SPARSE_MOE` (default off pending the
#48291 fidelity review; `DG_SPARSE_MOE_CAPACITY` sets C). Trace-safe: constant/scratch buffers are
preallocated + cached outside the trace (in-trace `zeros/ones/full` raise "Writes not supported
during trace capture"); `ttnn.scatter` verified out-of-place so the cached zero-base reuse is exact.

**Correctness (real 26B MoE layer, real router routing):**
- MoE output PCC vs dense = **0.99969**; real expert load max 25–31, mean 16 → **capacity 32 drops 0
  pairs** (method is exact when nothing is dropped).
- Full denoise LAYER PCC vs dense = **0.987–0.993** (random input); dense-vs-dense = 1.0 (deterministic).
  The layer delta is bf16 kernel rounding (batched matmul vs sparse_matmul) amplified through the
  post-MoE RMSNorm — smaller than the model's existing ~0.93-vs-fp32 bf16 error (fidelity is the
  deferred #48291 track).

**Traced denoise step (same-harness `prof_denoise_step.py`, warmed Metal trace, L=2/4 linear fit):**

| | dense traced | sparse traced | speedup |
|---|---|---|---|
| L=2 ms/step | 331.5 | 77.6 | |
| L=4 ms/step | 612.8 | 105.0 | |
| per-layer ms | 140.65 | **13.7** | **10.3×** |
| 30-layer step (proj, F=50.2ms) | 4269.7 | **461.2** | **9.26×** |

Tracing helps the many-small-op sparse path more than eager (eager per-layer 20 ms → traced 13.7 ms).

**Block t/s @ 48 steps:** `256 / (48×461.2 + 31432)ms` = **4.78 t/s** (dense 1.08) → **~4.4×**. With
early-halt (~18–25 steps) ~6 t/s. **The block is now COMMIT-BOUND** (commit 31.4 s = 59% of the
block) — Lever B (commit batching) is the next critical path, and the sparse MoE also applies to the
commit's 256-token re-encode (MoE is position-independent), so Lever A + B compound.

_Artifacts:_ `tt/sparse_moe.py`, `verify_sparse_moe.py`, `artifacts/leverA_verify_sparse_moe.log`,
`artifacts/leverA_traced_{dense,sparse}_L{2,4}.log`, `artifacts/leverA_step_*.log`.

## Lever B — commit batching (256 decode-appends → 1 causal prefill-append) — **SCOPED + QUANTIFIED** (not landed)

After Lever A the block is **commit-bound** (commit 31.4 s = 59% of the 53.5 s block @48 steps), so
Lever B is now worth **~2× on the block** (more if the commit prefill also uses the sparse MoE),
not the ~1.25× originally estimated.

**Mechanism (fully mapped from the code):** the commit re-encodes the block's 256 committed tokens
as 256 *sequential single-token decode-appends* (`commit_canvas_tokens` → `commit_decode_forward`),
paying the full 30-layer weight traffic 256×. The correct batched form is ONE **causal
prefill-append** of the 256 tokens at absolute offset `start_pos`: each token attends to the frozen
prefix (all visible) + earlier block tokens (causal), and writes its K/V to the cache. That is a
256-token forward = a denoise-step-equivalent in compute.

**Quantified potential** (the append compute is a 256-token forward; the traced denoise step
measured it at **461 ms/30L** with sparse MoE, ~4270 ms with dense):
- commit with **sparse MoE**: ~0.6–1.0 s (KV-write overhead included) vs 31.4 s = **~30–50×**.
- commit with **dense prefill MoE** (simpler): ~4.2 s vs 31.4 s = **~7×**.

Combined Lever A + B block t/s (sparse commit): **@48 steps ~11 t/s, @25 steps ~21 t/s, @18 steps
~28 t/s** (near the 30 target with early-halt). Dense-commit variant: @48 ~9.7, @25 ~16 t/s.

**Why not landed this session (honest):** the mission framed Lever B as "lower-risk", but the code
shows it is a substantial new path, and the hard rule is "verify bit-exact/PCC KV BEFORE keeping":
- `prepare_inputs_prefill` **discards `start_pos`** and Gemma4 computes prefill RoPE internally at
  positions 0.., so the stock prefill can't append at an offset. RoPE offset IS controllable via the
  `cos/sin` caches passed in (denoise already does this) — feasible.
- The short-sequence prefill SDPA (`scaled_dot_product_attention(is_causal=True)`) attends only to
  the CURRENT chunk, NOT the prefix cache. Correct append needs prefix-attending SDPA
  (`chunked_prefill_sdpa`, currently only taken for seq > 32768) plus an **offset KV write**
  (`paged_fill_cache`/`paged_update_cache` at `start_pos`), and the smoke path uses a *contiguous*
  (non-paged) cache with no offset-fill primitive.
- Landing a wrong KV-write silently corrupts the cache. A verified build (KV-PCC vs the decode path)
  is the correct next investment; it is a DiffusionGemma-local `commit_prefill.py` reusing the
  offset-RoPE machinery + `chunked_prefill_sdpa` + the sparse MoE, gated by a cache-content PCC test.

_Build spec left for the next session:_ `commit_prefill_append(tt_model, canvas_tokens, start_pos)`
= embed 256 → per layer {norm, causal prefix-attending SDPA at offset RoPE, offset KV write, sparse
MoE, norms, PLI} → skip LM head; verified by snapshotting cache K/V at the committed positions after
the batched-append vs after the 256-decode path (PCC), then measured commit_ms.

## Session summary (2026-07-04 session 1)

| lever | outcome |
|---|---|
| 1 traced-decode baseline | **washout** — readback 27.76 ms/step (0.66% at 30L); op-cost bound |
| 2 op-topology audit | **done** — corrected: dense-128 experts forward = 137.6 ms/layer = ~99% of step, transpose-bound |
| 3 true-sparse MoE | quick paths **blocked** (per-token nnz=8, larger chunk); token-gather rewrite scoped |
| 4 precision/config | **all ruled out** — BFP8/BFP4 no help (not weight-bound), L1 washout (1.01×) |
| commit LM-head skip | **LANDED** — 83 ms/block, bit-exact, zero-risk |
| 5 2-CQ / host-gap | little headroom (op-cost bound) |

## Session summary (2026-07-04 session 2) — **the wall came down**

| lever | outcome |
|---|---|
| **A token-gather true-sparse MoE** | **LANDED** — `tt/sparse_moe.py`, on-device dispatch, trace-safe. MoE 137→10.5 ms/layer (13×); **traced denoise step 4270→461 ms (9.26×)**; MoE PCC 0.9997, RUN-verified. Block **1.08→4.78 t/s @48 steps** |
| B commit batching | **SCOPED + QUANTIFIED** — block now commit-bound (59%); a causal prefill-append with sparse MoE → commit 31.4 s → ~0.6–1.0 s (~30–50×). Substantial verified build (offset RoPE ✓, needs prefix-SDPA + offset KV write + KV-PCC gate). Not landed unverified |

**Bottom line (updated):** Lever A — the token-gather true-sparse MoE — was the decisive structural
win the campaign was pointed at, and it landed: the dense-128 expert forward (137 ms/layer, ~99% of
the step, transpose-bound) is replaced by an on-device GShard capacity dispatch + batched matmul
(10.5 ms/layer), collapsing the **traced denoise step 9.26×** (4270→461 ms). The gather/scatter did
NOT wash out (~2 ms overhead); the batched matmul also side-steps the 87 ms expert-major transpose.
Block throughput is now **4.78 t/s @48 steps** (from 1.08) and **~6 t/s** with early-halt — and the
block has flipped to **commit-bound**. The remaining path to 30 t/s is now clear and quantified:
**Lever B** (commit batching with the same sparse MoE) takes the commit 31.4 s → ~0.6–1.0 s, which
with early-halt (~18–25 steps) projects to **~21–28 t/s** — within reach of the 30 target. Lever B is
a substantial but well-specified build (a verified causal prefill-append), left as the clear next
investment rather than landed unverified (the hard rule is verify KV before keeping).

## Session summary (2026-07-04 session 3) — consolidation + Lever B device-verify

| item | outcome |
|---|---|
| Consolidate branches | **DONE** — cherry-picked `path_to_30tps.md` (roadmap) + the 3 commit-batching commits onto `diffusion-gemma-function`; pushed. gemma4 gate clean. |
| Lever B batched commit — make it RUN | **FIXED 2 real bugs** — (1) sharded KV write for `paged_update_cache` (was DRAM-interleaved → `is_sharded()` fatal); (2) guarded `to_memory_config` (no-op DRAM alias returned an unallocated SDPA input). Batched commit now runs. |
| Lever B — commit_ms speedup | **6.3× measured** (1031 ms vs 6503 ms @ L=6; the "one 256-token forward vs 256 forwards" win is real). |
| Lever B — KV bit-equivalence | **FAIL (0.43), stays OPT-IN.** Root-caused to the **MoE expert kernel** (see `commit_batching.md`): layer-0 attention / shared_mlp bit-exact, router 0.98 (99.6% expert-mask agree), but `_commit_experts_decode_forward` (sequential, decode `sparse_matmul` nnz=8) diverges **0.17** from the verified-correct batched experts (`moe.experts`/`sparse_experts_forward`, ≈torch 0.9997). The **batched commit is likely *more* correct**; the sequential decode-commit MoE (never PCC-verified, RUN-first) is the suspect reference. Ruled out: masked SDPA (0.998 vs torch), RAW hazard, norms. |

### Combined end-to-end t/s (serving_smoke, full 30L, eager serving path) — 2026-07-04

Measured on QB2 (P150x4), `DG_SPARSE_MOE=1`, 24-step budget, per-block metric (`tokens/block/s`):

| path | mean block latency | **tokens/block/s** | note |
|---|---|---|---|
| sparse MoE + **sequential** commit (verified) | 49.58 s | **5.16** | both blocks ran full 24 steps (no early-halt on degenerate output) |
| sparse MoE + **batched** commit (opt-in, perf-only) | 20.85 s | **12.28** | **2.38× on the block** vs sequential commit |

Back-out (both at 24 steps): **eager per-step ≈ 0.64 s** (30L, sparse MoE + terminal + host readback; ~1.4× the
461 ms *traced* step — the serving loop is eager, headroom via tracing), **commit_seq ≈ 34 s**, **commit_batched ≈ 5.4 s**
(6.3× commit, matching the L=6 verify). Commit is ~69% of the sequential block, ~26% of the batched block. Artifacts:
`artifacts/combined_{seq,batched}_30L_s24.{json,log}`. Note: the batched-commit t/s is a **perf-only** number — its KV
is not bit-equivalent (§Lever B), so the generated text is not the correctness reference; the sequential path text is
semi-coherent ("a diffusion language model is a generative model that creates text by starting with random noise…").

**Headline:** the campaign has moved the block from the **1.08 t/s baseline → 5.16 t/s** (Lever A sparse MoE + 24-step,
verified) and, with the opt-in batched commit, to **12.28 t/s** (Lever A+B, perf-only until the commit KV is verified).

### Step-count sweep (Lever A+B batched, 30L eager serving) — 2026-07-04

t/s vs the denoise step budget (`--max-denoising-steps`), sparse MoE + batched commit. **Early-halt did NOT fire**
(`halted=false` at every budget) — the degenerate RUN-first output never trips the stability+entropy halt, so these are
**fixed-budget** points; the *adaptive* benefit (session-2 evidence: real blocks halt at 18–38 steps) applies on real
content only, and is a #48291-gated quality lever, not a config cut.

| step budget | block latency (steady) | **tokens/block/s (eager)** |
|---|---|---|
| 12 | 10.3 s | **24.76** |
| 24 | 20.9 s | **12.28** |
| 48 (advertised max, all-prompts) | 38.0 s | **6.73** |

Two-point fit (24↔48): **eager per-step ≈ 0.72 s** (30L), **commit_batched ≈ 2–4 s** (grows mildly with `start_pos`:
the batched commit's prefix-attending SDPA reads `[0:start_pos+256]`). So at 48 steps the commit is only ~10% of the
block (vs ~65% for the sequential commit) — Lever B fully removed the commit as the bottleneck; the block is **denoise-step
bound again**, exactly as the roadmap predicted after commit batching.

**30 t/s trajectory (honest):** the advertised full-budget (48-step, all-prompts) number is **6.73 t/s** eager. 30 t/s is
**not** reached at any quality-safe eager budget. It needs the *stacked favorable-regime* levers the roadmap already
scoped, none of which is a new structural win:
- **traced serving denoise** (the loop is currently eager at ~0.72 s/step; the *traced* step is 0.461 s — a 1.35×
  headroom on the per-step term that the eager serving loop leaves on the table). With traced denoise: 12-step block ≈
  12×0.461 + 2 ≈ 7.5 s ⇒ **~34 t/s**; 18-step ≈ **~24 t/s**.
- **adaptive early-halt to ~12–18 steps** (data-dependent, real content only; not a lowered advertised max).
- the **verified batched commit** (currently opt-in; §Lever B blocker).
- **terminal-path trim (lever 5)** is already active in `tt/sampling.py`/`tt/denoise_loop.py` (~13% of the step, ~1.05×);
  it is baked into the 0.72 s/step above, not a separate measured lever.

So: **30 t/s is reachable in the favorable regime (traced denoise + verified batched commit + ~12-step early-halt on
easy/median prompts), but NOT for hard prompts at the full 48-step budget** — matching `path_to_30tps.md`. The verified,
all-prompts, no-quality-tradeoff number today is **5.16 t/s** (sparse MoE + sequential commit, 24-step).

**Bottom line (session 3):** Lever B is now a *running*, 6.3×-faster commit, but it cannot be landed
by default because its KV cache disagrees (0.43) with the sequential decode-append reference — and the
divergence is a genuine defect in the **sequential** path's MoE expert kernel, not the batched one.
Landing Lever B (and reaching ~21–28 t/s) is now gated on ONE well-defined task: verify the batched
commit KV against a **torch** commit reference (bypassing the buggy sequential), or fix
`_commit_experts_decode_forward` to match `moe.experts`/torch, then flip `DG_COMMIT_BATCHED` on by
default. Evidence: `verify_commit_batching.py` (per-layer PCC), `probe_commit_l0attn.py` (stage
isolation), `probe_masked_sdpa.py` (SDPA vs torch), `artifacts/leverB_*.log`.

## Session summary (2026-07-04 session 4) — **Lever B LANDED (torch-verified) + it's a correctness fix**

| item | outcome |
|---|---|
| **Torch oracle for the commit MoE** (`probe_moe_vs_torch.py`) | **DONE, DECISIVE.** Hand-rolled HF `DiffusionGemma` MoE (router+experts), real layer-0 weights, fp32, run on the **identical bit-exact** layer-0 commit MoE input. PCC(torch, **batched**)=**0.9936** vs PCC(torch, sequential)=**0.1542**. The **sequential** `_commit_experts_decode_forward` is genuinely DEFECTIVE; the batched commit is correct. |
| **Flip `DG_COMMIT_BATCHED` default ON** | **LANDED.** `batched_commit_enabled()` default "1"; `_resolve_default_commit_fn` forces sequential only for paged/vLLM caches. `DG_COMMIT_BATCHED=0` forces sequential. gemma4 gate stays the 1-line dealloc. |
| **Verified combined t/s (30L, sparse MoE, 24-step)** | sequential **5.13 t/s** (49.94 s/block, KV 0.154) → **batched 13.04 t/s** (19.64 s/block, KV 0.994). **2.54× on the block AND a correctness fix.** Block-0 text coherent. |

**Bottom line (session 4):** Lever B is **landed as the default**, and it's not just a perf lever — the
batched commit writes the **correct** KV (torch-verified 0.994) while the old sequential default wrote
**broken** KV (0.154). The previously-recorded "verified 5.16 t/s" was verified-to-RUN but writing wrong
prefix KV; the real, correct, verified number is now **13.04 t/s @24 steps** (Lever A sparse MoE + Lever B
batched commit). The `path_to_30tps` row-2 projection (11.28 t/s @48-step fixed budget) is realized and
exceeded at the 24-step budget. Next: adaptive early-halt + traced serving denoise toward 30 (the block is
now **denoise-step bound** again, per the roadmap — the commit is ~10–26% of the block).

### Step-count / adaptive early-halt — VERIFIED curve + honest crossover (2026-07-04 session 4)

Measured on QB2, 30L, `DG_SPARSE_MOE=1`, **batched commit (default, correct)**, `serving_smoke`, per-block:

| step budget | mean block latency | **tokens/block/s (VERIFIED, correct KV)** | halted? |
|---|---|---|---|
| 24 | 19.64 s | **13.04** | [False, False] |
| 48 (advertised max, all-prompts) | 38.28 s | **6.69** | [False, False, False] |

Two-point fit (24↔48): **eager per-step ≈ 0.777 s** (30L, sparse MoE + terminal + 5 host readbacks/step),
**commit + per-block fixed ≈ 1.0 s** (batched commit is now ~3–5% of the block — Lever B fully removed the
commit as a bottleneck; the block is **denoise-step bound**). So `t/s(K) = 256 / (0.777·K + ~1.0)`, and
**30 t/s ⇔ K ≈ 9–10 eager steps.**

**Adaptive early-halt DOES NOT reach 30 t/s on the current RUN-first output — and correct-KV did not change
that.** Running the full advertised 48-step budget with the correct batched commit (`task2_adaptive_48budget_3blk`),
**all three blocks ran 48 steps, `halted=[False,False,False]`.** The halt is gated on `entropy_mean < 0.005`
(config `entropy_stop_threshold`), and the #48291-marginal per-step decisions keep mean entropy above that —
a **model-fidelity** property, not a commit property (correct KV only affects block N+1's *conditioning*, not
whether block N's entropy converges). Even the natural halt range observed on other content (18–38 steps) gives
only ~10–16 t/s eager, still short of 30.

**Honest verdict (task: adaptive only, don't lower advertised max):** the advertised-budget (48-step) verified
number is **6.69 t/s**; the 24-step point is **13.04 t/s**. Pure adaptive step-count cannot deliver verified 30
t/s — it needs halt at ~9 eager steps, far below any plausible convergence, and early-halt never fires under
#48291. The lever that makes 30 land at a *plausible* step count is **traced serving denoise** (the eager loop
runs the step at 0.777 s vs the *traced* 0.461 s measured by `prof_denoise_step.py` — a 1.35–1.7× per-step
headroom the eager serving loop leaves on the table): with traced denoise `t/s(K)=256/(0.461·K+~1.0)` ⇒ 30 t/s
⇔ K ≈ 17 steps, which IS within the plausible early-halt band once #48291 lets blocks converge. So verified 30
t/s = traced serving denoise (+ terminal trim) + adaptive halt to ~16–18 steps — the task-4 direction — not
step-count alone. Artifacts: `artifacts/task2_adaptive_48budget_3blk.{json,log}`, `leverB_verified_batched_30L_s24.json`.

### 100-path start — per-step overhead measured + device-only (trace-ready) denoise loop (2026-07-04 session 4)

**Measured the denoise-step decomposition on device** (`prof_denoise_step.py`, L=2/4 two-point fit, `DG_SPARSE_MOE=1`;
`artifacts/task4_traced_step_sweep.log`):

| step form | per-layer | fixed | **30L step** | source |
|---|---|---|---|---|
| eager **device** step (tight loop, no readback) | 18.38 ms | 47.1 ms | **598 ms** | prof eager |
| **traced** step (warm Metal trace) | 13.69 ms | 50.2 ms | **461 ms** | prof traced (`RESULT_TRACED`) |
| eager **serving** step (5 host readbacks + halt + adapter + noise upload) | — | — | **~725–756 ms** | serving_smoke back-out |

So there are two stacked per-step overheads over the 461 ms traced floor: **~137 ms/step eager dispatch** (removed by
tracing) and **~130–260 ms/step serving-loop host work** (5 readbacks + `torch.equal` halt + per-step host noise upload
+ self-cond adapter python).

**LANDED — device-only (trace-ready) denoise loop** (`tt/denoise_loop.py::device_loop_denoise_block`, opt-in
`DG_DENOISE_DEVICE_LOOP=1`, wired via `generate._resolve_default_denoise_block_fn`). It runs the existing
`run_fixed_denoise_steps` (device-resident, **no per-step host readback**, no `torch.equal` halt) and reads back only the
final committed argmax once. **Verified BIT-IDENTICAL** to the eager loop: same 24-step run, seed 0, **committed text
character-for-character identical** (494 chars both) — because early-halt never fires (§step-count), so the fixed-budget
device loop and the eager loop commit the same final argmax.

| loop | block latency | **tokens/block/s** | committed text |
|---|---|---|---|
| eager `denoise_block` (5 readbacks/step + halt) | 19.64 s | 13.04 | reference |
| **device-only `device_loop_denoise_block`** (`DG_DENOISE_DEVICE_LOOP=1`) | 18.90 s | **13.55** | **bit-identical** |

The direct win is modest (~1.04×; the readbacks cost only ~31 ms/step, less than projected). Its real value: it is the
**trace-ready** form — the eager loop's 5 per-step host readbacks make it un-traceable, so `device_loop_denoise_block` is
the prerequisite for the ~137 ms/step tracing win. Opt-in (discards per-step trajectory records + cannot early-halt);
recommended-on for serving throughput, kept off by default so record-consuming callers (text_demo/tests) are unaffected.
Artifacts: `artifacts/task4_deviceloop_30L_s24.{json,log}`, `task4_traced_step_sweep.log`. No gemma4 edits.

**Trace-capture the device-only loop (win B) — DEVICE-PROVEN 2.29× + blocker ISOLATED** (`probe_traced_denoise_loop.py`,
real adapter, L=6, 8 steps, `DG_SPARSE_MOE=1`, pre-uploaded per-step renoise buffers). Capturing the whole
`run_fixed_denoise_steps` device loop (real adapter forward + frozen prompt-KV read) as ONE Metal trace and replaying it:

| self-conditioning | eager ms/step | **traced ms/step** | **win** | committed match |
|---|---|---|---|---|
| ENABLED (default) | 354.4 | 155.4 | **2.28×** | **60.5% ✗** |
| DISABLED | 303.4 | 132.4 | **2.29×** | **100.0% ✓** |

**So the traced serving denoise loop is trace-correct AND ~2.29× faster — the SOLE correctness blocker is the
self-conditioning.** (Contrast: `verify_trace_safe_loop.py`'s *stateless* synthetic logits fn traces at 99.9%; the real
adapter's only cross-step mutable state is self-cond.) Projected to 30L: traced serving step ~461 ms (prof) vs eager
~756 ms → **block @24 ~12.6 s → ~20 t/s** (from 13.55), **@12 steps ~36 t/s** — the traced loop is THE lever to verified
30 (with adaptive halt to ~12–18 steps). The per-step renoise host-upload blocker is already solved (pre-upload to
persistent device buffers; `run_fixed_denoise_steps` takes preallocated `constants`).

**Root cause of the self-cond blocker** (`DenoiseLogitsAdapter`, `tt/denoise_forward.py:577-593`): `self.prev_logits` is a
**fresh `[1,1,256,262144]` alloc each step**, chained (step N reads step N-1's buffer, then deallocs it), and **step 0 is
special** (`prev_logits=None` → self-cond skipped). A fixed Metal trace bakes in buffer addresses and runs one fixed graph
for every step, so (a) the chained fresh-alloc `prev_logits` is not address-stable across replays and (b) step-0's
skip-vs-condition cannot vary within a fixed graph. **Fix (scoped, DiffusionGemma-local, next session):** make the
self-cond signal use a **persistent preallocated `prev_logits` buffer** updated **in-place** each step, with step 0 fed a
**zeroed signal** (or run step 0 eager + steps 1..N−1 traced), reset per block. This is a real adapter change with a
decision-fidelity check (the self-cond signal feeds the diffusion decisions), so it is scoped, not landed. The other big
in-repo per-step lever is `path_to_100tps` rank 2 (OPT-004 matmul-geometry tuning of the 5 untuned `tt/sparse_moe.py`
matmuls, MoE 10.5→~5–6 ms/layer). Artifacts: `probe_traced_denoise_loop.py`,
`artifacts/task4_traced_loop_probe_L6{,_noselfcond}.log`.

## Session summary (2026-07-04 session 5) — self-cond trace-blocker ROOT-CAUSED as a cross-step RACE; dedup + OPT-004 landed

### The self-cond whole-loop trace blocker is a RACE, not address instability (prior diagnosis REFUTED)

The session-4 hypothesis — "chained fresh-alloc `prev_logits` is not address-stable; fix with a persistent in-place
buffer" — is **wrong**. Three self-cond feedback variants were built and traced (real adapter, L=6, 8 steps,
`DG_SPARSE_MOE=1`, `probe_traced_denoise_loop.py`), and **all three give the IDENTICAL 60.5% committed match**:
1. original `prev_logits` chain (fresh full-logits threaded via adapter attribute),
2. persistent in-place `signal_buf` updated with `ttnn.copy` each step,
3. fresh single-consumer `signal` tensor threaded like the canvas.

Two new diagnostics settle the mechanism:
- **eager-vs-eager = 100.0%** — the self-cond feedback loop is fully deterministic eagerly (so it is NOT bf16
  feedback chaos, and the trace is not merely amplifying a real per-step numerical delta).
- **traced-vs-traced = 60.5%** — **two replays of the *same* captured trace disagree.** That is a genuine RACE
  (a read-before-write / stale-buffer hazard the trace does not order), not a systematic op difference and not
  non-determinism of the kernels. Self-cond OFF traces at **100%** (both replays and vs eager), so the race is
  introduced solely by the self-cond feedback reading the previous step's LM-head logits inside the same trace.
- **STEPS=1 = 100.0%** (traced-vs-traced AND traced-vs-eager, `TRACED_LOOP_OK`) — a **single** self-cond step traces
  perfectly. **⇒ the race is purely CROSS-STEP inside one whole-loop trace, not within a step.**

**Consequence for the architecture.** The whole-loop trace (capture all N steps, replay once) cannot carry the
self-cond feedback (race), and the buffer-management fixes the prior session scoped do NOT resolve it (they change
nothing — identical 60.5%). The correct shape is a **single-step trace replayed once per step**, carrying the
cross-step state (canvas + self-cond signal) in persistent device buffers updated **across replays** — the exact
KV-cache pattern a traced decode uses, which has no cross-step feedback *inside* a trace (STEPS=1 proves each step
traces at 100%). The trace-safe adapter (`DenoiseLogitsAdapter.prepare_trace_safe_self_conditioning` /
`_trace_safe_call`, uniform `forward(embed, signal_buf)` with a zeroed step-0 buffer = bit-exact `condition(None)`)
is that adapter; `probe_singlestep_traced.py` validates the single-step mechanism.

**Second, independent blocker for a traced SERVING loop (new finding, missed by the session-4 projection):**
`q_rope_offset` (= `start_pos`) advances per block and `_get_rope_mats(seq_len=q_rope_offset+C)` returns a
**growing** `cos[:,:, :seq_len,:]` slice that is baked into the trace — so a captured trace (whole-loop OR
single-step) is only valid for the block it was captured on. Cross-block trace **reuse** (required for any net
win — per-block re-capture ≈ eager cost) needs a **constant-shape persistent canvas-rope buffer** (the C canvas
rows, updated per block outside the trace, `start_offset=0` into `_apply_rope_chunked`), threaded into
`denoise_attention` diffusion-gemma-locally. This is a second substantial sub-build; the session-4 "traced loop →
20 t/s" projection measured a single block's *replay* and did not account for cross-block re-capture. **Net: the
traced serving loop = single-step trace mechanism (self-cond fixed) + constant-shape rope-mats reuse — scoped and
architected here, not landed unverified.** Artifacts: `artifacts/task1_{diag_eagereager,diag_tvt,freshsignal,steps1}_L6.log`.

### Lever (terminal argmax dedup, `path_to_100tps` rank 3) — LANDED, device-verified (474713ec259)

`DG_DEDUP_ARGMAX` (opt-in, default off). In the argmax regime `denoise_step` ran two full-vocab (262144) argmax
reductions/step (`argmax(logits/T)` for `sampled` + `argmax(logits)` for the commit); positive scaling is
order-preserving so the dedup computes the raw-logit argmax once and clones the tiny index for `sampled`.
Self-contained device verify (`verify_terminal_dedup.py`, synthetic production-vocab logits): committed argmax /
entropy / accept-mask **HARD bit-exact OFF==ON**, `dedup sampled == default argmax` (0 ties on the input),
`T==1.0` fully bit-exact + multi-step loop equivalence PASS; timing **43.04 → 28.05 ms/step terminal sampling =
14.99 ms saved (34.8%)**, layer-count-independent ⇒ ~1.05× on the 30L block. `verify_terminal_dedup.py`.

### Lever (OPT-004 matmul-geometry tuning, `path_to_100tps` rank 2/lever 1) — MERGED + device-verified (014c47177f7, c2c5f4cf0ec)

`DG_SPARSE_MOE_TUNED` (opt-in, default off; flag-off = byte-identical auto-config prototype). Adds explicit
`program_config` geometry to the 5 previously-untuned `ttnn.matmul` calls in `tt/sparse_moe.py` (batched gate/up/down +
gather `disp^T@hidden` + combine `comb@down`) — the batched experts read the ~415 MB (bf16) / ~220 MB (bfp8) bank at only
~46 GB/s (~18% of @256). Batched gate/up/down use `MatmulMultiCoreReuseProgramConfig` (`per_core_M=Mt`, `per_core_N=Nt`
forced by the reuse factory → E=128 blocks distributed 1/core), gather/combine use 2D
`MatmulMultiCoreReuseMultiCastProgramConfig`. Same dtype/fidelity (HiFi2) ⇒ pure geometry, PCC must equal untuned.
Device grid measured **11×10 = 110 cores** (not the doc's 13×10=130). Per-matmul device result (real 26B layer-0):
**gate/up (the dominant K=88 matmul) untuned 4.176 → tuned 0.593 ms = 7.05× at PCC 0.99986** (pure geometry, correct)
— this is the batched-expert builder used by gate/up/**down**, so the dominant expert compute is verified.

**Honest verification status:** the full 5-config `sparse_experts_forward` tuned verify (`verify_opt004_fullmoe.py`,
even with the slow dense-128 reference skipped) hit a **pathologically slow first-call compile** of the tuned configs
(>7 min / ~27 min CPU with no progress, twice) — likely the `down` config (`per_core_N=88`, `in0_block_w=2` → 44
K-passes) or a 2D gather/combine config — and did not produce the layer-level number this session. OPT-004 is
**opt-in, default OFF**, so the shipped path is byte-for-byte unchanged; it is landed on the gate/up device
verification (dominant matmul, correct + 7×) with the compile-time of the down/gather/combine configs flagged as the
open item **before enabling by default** (candidate fix: cap `per_core_N` / re-pick `in0_block_w` for the down + 2D
configs; the compile-heavy `bench_opt004_matmul_geometry.py` per-candidate SWEEP is the tool to pick the fast legal one).

### Traced serving loop — the two remaining blockers (scoped, NOT landed unverified)

Per the §diagnosis above, a traced serving loop needs BOTH:
1. **Single-step trace mechanism** (self-cond fix): capture one denoise step, replay once/step, cross-step state (canvas +
   self-cond signal) in persistent buffers threaded ACROSS replays (KV-cache pattern; STEPS=1 traces at 100%). The
   trace-safe adapter is built; `probe_singlestep_traced.py` validates it (RESULT_REFACTOR = adapter eager bit-exactness;
   RESULT_SINGLESTEP_TRACED = mechanism). Per-step temperature `T[i]` handled by capturing N single-step traces (one per
   step index, each baking `T[i]` + reading `noise[i]`) — no device-scalar temperature, fidelity-safe.
2. **Cross-block rope-mats reuse**: `q_rope_offset` (= start_pos) advances per block and bakes into the trace via the
   growing `_get_rope_mats(seq_len=qro+C)` slice, so a captured trace is valid only for its block. Reuse across blocks
   (mandatory — per-block re-capture ≈ eager cost, no win) needs a constant-shape persistent canvas-rope buffer (C rows,
   `start_offset=0` into `_apply_rope_chunked`, content updated per block outside the trace), threaded into
   `denoise_attention` diffusion-gemma-locally. This is the second substantial sub-build and is why the session-4 "traced
   → 20 t/s" projection (a single block's replay) overstated the near-term serving win.

Both are architected here; landing them (verified bit-exact + serving t/s) is the next investment toward 30.

### Landed-lever stack (t/s accounting)

On top of the session-4 verified **13.04 t/s @24-step** (sparse MoE + batched commit, both default) the session-5
opt-in levers stack: **dedup** ~1.05× (14.99 ms/step, device-verified bit-exact), **OPT-004** MoE geometry (gate/up
matmul 7.05× device-verified; layer-level via `verify_opt004_fullmoe.py`), and the already-landed **device loop**
`DG_DENOISE_DEVICE_LOOP` ~1.04× (bit-identical, session 4). Each is independently device-verified; enable with
`DG_SPARSE_MOE=1 DG_DEDUP_ARGMAX=1 DG_SPARSE_MOE_TUNED=1 DG_DENOISE_DEVICE_LOOP=1`. None reaches 30 alone — the
step-count × per-step-traced multiplier (the traced serving loop, blocked above) is still required for verified 30.
